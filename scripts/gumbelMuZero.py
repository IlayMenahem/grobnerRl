"""
Gumbel MuZero implementation for the Buchberger environment.

This module implements the Gumbel MuZero algorithm which uses Gumbel-Top-k
sampling with Sequential Halving for efficient action selection without
requiring full tree search.

References:
    - "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)
"""

import os
from copy import copy
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.models import (
    Extractor,
    GrobnerPolicy,
    IdealModel,
    MonomialEmbedder,
    PairwiseScorer,
    PolynomialEmbedder,
)
from grobnerRl.training.utils import load_checkpoint, save_checkpoint
from grobnerRl.types import Observation


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    monomials_dim: int
    monoms_embedding_dim: int = 64
    polys_embedding_dim: int = 128
    ideal_depth: int = 4
    ideal_num_heads: int = 8
    value_hidden_dim: int = 128


@dataclass
class GumbelMuZeroConfig:
    """Configuration for Gumbel MuZero search."""

    num_simulations: int = 16
    max_num_considered_actions: int = 16
    gamma: float = 0.99
    c_visit: float = 50.0
    c_scale: float = 1.0


@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs_per_iteration: int = 3
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


class GumbelMuZeroModel(Module):
    """
    Neural network model for Gumbel MuZero with policy and value heads.

    The model uses a shared backbone (Extractor) for feature extraction
    and separate heads for policy logits and value estimation.
    """

    extractor: Extractor
    value_head: eqx.nn.MLP

    def __init__(self, extractor: Extractor, value_head: eqx.nn.MLP):
        self.extractor = extractor
        self.value_head = value_head

    def __call__(self, obs: Observation | dict | tuple) -> tuple[Array, Array]:
        """
        Forward pass returning policy logits and value estimate.

        Args:
            obs: Environment observation (tuple or dict format).

        Returns:
            Tuple of (policy_logits, value).
        """
        policy_logits = self.extractor(obs)

        if isinstance(obs, dict):
            ideal_stacked = obs["ideals"]
            masks_stacked = obs["monomial_masks"]
            poly_mask = obs["poly_masks"]

            monomial_embs = jax.vmap(self.extractor.monomial_embedder)(ideal_stacked)
            ideal_embeddings = jax.vmap(self.extractor.polynomial_embedder)(
                monomial_embs, masks_stacked
            )
            ideal_embeddings = self.extractor.ideal_model(
                ideal_embeddings, mask=poly_mask
            )

            masked_embs = jnp.where(poly_mask[:, None], ideal_embeddings, 0.0)
            pooled = masked_embs.sum(axis=0) / (poly_mask.sum() + 1e-9)
        else:
            ideal, _ = obs

            ideal_arrays = [jnp.asarray(p) for p in ideal]
            lengths = [p.shape[0] for p in ideal_arrays]
            max_len = max(lengths) if lengths else 1

            padded_ideal = []
            masks = []
            for p in ideal_arrays:
                length = p.shape[0]
                pad_len = max_len - length
                if pad_len > 0:
                    p_padded = jnp.pad(p, ((0, pad_len), (0, 0)), constant_values=0)
                    mask = jnp.concatenate(
                        [jnp.ones(length, dtype=bool), jnp.zeros(pad_len, dtype=bool)]
                    )
                else:
                    p_padded = p
                    mask = jnp.ones(length, dtype=bool)
                padded_ideal.append(p_padded)
                masks.append(mask)

            ideal_stacked = jnp.stack(padded_ideal)
            masks_stacked = jnp.stack(masks)

            monomial_embs = jax.vmap(self.extractor.monomial_embedder)(ideal_stacked)
            ideal_embeddings = jax.vmap(self.extractor.polynomial_embedder)(
                monomial_embs, masks_stacked
            )

            poly_mask = jnp.ones(ideal_embeddings.shape[0], dtype=bool)
            ideal_embeddings = self.extractor.ideal_model(
                ideal_embeddings, mask=poly_mask
            )

            pooled = ideal_embeddings.mean(axis=0)

        value = self.value_head(pooled).squeeze(-1)

        return policy_logits, value

    @classmethod
    def from_scratch(cls, config: ModelConfig, key: Array) -> "GumbelMuZeroModel":
        """Initialize model from scratch."""
        keys = jax.random.split(key, 5)
        k_monomial, k_polynomial, k_ideal, k_scorer, k_value = keys

        monomial_embedder = MonomialEmbedder(
            config.monomials_dim, config.monoms_embedding_dim, k_monomial
        )
        polynomial_embedder = PolynomialEmbedder(
            input_dim=config.monoms_embedding_dim,
            hidden_dim=config.polys_embedding_dim,
            hidden_layers=2,
            output_dim=config.polys_embedding_dim,
            key=k_polynomial,
        )
        ideal_model = IdealModel(
            config.polys_embedding_dim,
            config.ideal_num_heads,
            config.ideal_depth,
            k_ideal,
        )
        pairwise_scorer = PairwiseScorer(
            config.polys_embedding_dim, config.polys_embedding_dim, k_scorer
        )
        extractor = Extractor(
            monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer
        )

        value_head = eqx.nn.MLP(
            in_size=config.polys_embedding_dim,
            out_size=1,
            width_size=config.value_hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=k_value,
        )

        return cls(extractor=extractor, value_head=value_head)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: ModelConfig,
        optimizer: optax.GradientTransformation,
        key: Array,
    ) -> "GumbelMuZeroModel":
        """Initialize from a pretrained GrobnerPolicy checkpoint."""
        key, k_value, k_load = jax.random.split(key, 3)

        k_monomial, k_polynomial, k_ideal, k_scorer = jax.random.split(k_load, 4)

        monomial_embedder = MonomialEmbedder(
            config.monomials_dim, config.monoms_embedding_dim, k_monomial
        )
        polynomial_embedder = PolynomialEmbedder(
            input_dim=config.monoms_embedding_dim,
            hidden_dim=config.polys_embedding_dim,
            hidden_layers=2,
            output_dim=config.polys_embedding_dim,
            key=k_polynomial,
        )
        ideal_model = IdealModel(
            config.polys_embedding_dim,
            config.ideal_num_heads,
            config.ideal_depth,
            k_ideal,
        )
        pairwise_scorer = PairwiseScorer(
            config.polys_embedding_dim, config.polys_embedding_dim, k_scorer
        )
        extractor = Extractor(
            monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer
        )
        template_policy = GrobnerPolicy(extractor)

        template_opt_state = optimizer.init(eqx.filter(template_policy, eqx.is_array))

        template = {
            "model": template_policy,
            "opt_state": template_opt_state,
            "epoch": 0,
            "val_accuracy": 0.0,
        }

        payload = load_checkpoint(checkpoint_path, template)
        pretrained_extractor = payload["model"].extractor

        value_head = eqx.nn.MLP(
            in_size=config.polys_embedding_dim,
            out_size=1,
            width_size=config.value_hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=k_value,
        )

        return cls(extractor=pretrained_extractor, value_head=value_head)


@dataclass
class GumbelNode:
    """
    A node in the Gumbel MuZero search tree.

    Attributes:
        env: Copy of the environment at this state (None for unexpanded children).
        visit_count: Number of visits to this node.
        value_sum: Sum of backed-up values.
        reward: Immediate reward received when reaching this node.
        prior: Prior probability from policy network.
        is_terminal: Whether this is a terminal state.
        children: Dictionary mapping actions to child nodes.
    """

    env: BuchbergerEnv | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    children: dict[int, "GumbelNode"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def sigma(logits: np.ndarray, q_values: np.ndarray, visit_counts: np.ndarray,
          c_visit: float, c_scale: float) -> np.ndarray:
    """
    Compute the completed Q-values using sigma transformation.

    This transforms raw Q-values into a scale compatible with policy logits
    for action selection.

    Args:
        logits: Policy logits from neural network.
        q_values: Q-values for each action.
        visit_counts: Visit counts for each action.
        c_visit: Visit count scaling constant.
        c_scale: Q-value scaling constant.

    Returns:
        Transformed Q-values (sigma values).
    """
    max_logit = logits.max()
    normalized_visits = visit_counts.sum() / (visit_counts.sum() + c_visit)
    
    max_q = q_values.max() if visit_counts.sum() > 0 else 0.0
    min_q = q_values.min() if visit_counts.sum() > 0 else 0.0
    q_range = max_q - min_q
    
    if q_range > 0:
        scale = c_scale * q_range
    else:
        scale = c_scale
    
    sigma_values = normalized_visits * (max_logit + scale * q_values)
    return sigma_values


def sample_gumbel(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    """Sample from Gumbel(0, 1) distribution."""
    u = jax.random.uniform(key, shape, minval=1e-10, maxval=1.0 - 1e-10)
    return -jnp.log(-jnp.log(u))


def gumbel_top_k(
    key: PRNGKeyArray,
    logits: np.ndarray,
    valid_actions: list[int],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample top-k actions using Gumbel-Top-k trick.

    Args:
        key: JAX random key.
        logits: Policy logits.
        valid_actions: List of valid action indices.
        k: Number of actions to sample.

    Returns:
        Tuple of (selected_actions, gumbel_values).
    """
    valid_logits = logits[valid_actions]
    gumbel_noise = np.array(sample_gumbel(key, (len(valid_actions),)))
    
    perturbed = valid_logits + gumbel_noise
    
    k = min(k, len(valid_actions))
    top_k_indices = np.argpartition(perturbed, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(perturbed[top_k_indices])[::-1]]
    
    selected_actions = np.array([valid_actions[i] for i in top_k_indices])
    gumbel_values = gumbel_noise[top_k_indices]
    
    return selected_actions, gumbel_values


def sequential_halving(
    actions: np.ndarray,
    gumbel_values: np.ndarray,
    root: GumbelNode,
    env_template: BuchbergerEnv,
    model: GumbelMuZeroModel,
    config: GumbelMuZeroConfig,
) -> int:
    """
    Sequential Halving procedure for action selection.

    Iteratively reduces the action set by half, allocating simulations
    to remaining actions and eliminating the worst half.

    Args:
        actions: Initial set of candidate actions.
        gumbel_values: Gumbel values for each action.
        root: Root node of the search tree.
        env_template: Environment template for simulation.
        model: Neural network model.
        config: Gumbel MuZero configuration.

    Returns:
        Selected action index.
    """
    if len(actions) == 1:
        return int(actions[0])

    assert root.env is not None, "Root node must have an environment"
    num_polys = len(root.env.generators)
    remaining_actions = actions.copy()
    remaining_gumbels = gumbel_values.copy()

    num_phases = int(np.ceil(np.log2(len(actions))))
    sims_per_phase = max(1, config.num_simulations // num_phases)

    for phase in range(num_phases):
        if len(remaining_actions) <= 1:
            break

        sims_this_phase = sims_per_phase // len(remaining_actions)
        sims_this_phase = max(1, sims_this_phase)

        for _ in range(sims_this_phase):
            for action in remaining_actions:
                action = int(action)
                if action not in root.children or root.children[action].env is None:
                    child_env = copy_env(root.env)
                    i, j = action // num_polys, action % num_polys
                    _, reward, terminated, _, _ = child_env.step((i, j))

                    if terminated:
                        child_value = 0.0
                    else:
                        obs = make_obs(child_env.generators, child_env.pairs)
                        _, child_value = model(obs)
                        child_value = float(child_value)

                    existing_prior = (
                        root.children[action].prior if action in root.children else 0.0
                    )
                    root.children[action] = GumbelNode(
                        env=child_env,
                        reward=reward,
                        is_terminal=terminated,
                        prior=existing_prior,
                    )

                child = root.children[action]
                child.visit_count += 1
                child.value_sum += child.reward + config.gamma * child.q_value

        q_values = np.array([
            root.children[int(a)].q_value if int(a) in root.children else 0.0
            for a in remaining_actions
        ])
        visit_counts = np.array([
            root.children[int(a)].visit_count if int(a) in root.children else 0
            for a in remaining_actions
        ])

        logits = np.zeros(len(remaining_actions))
        sigma_values = sigma(logits, q_values, visit_counts, config.c_visit, config.c_scale)

        scores = remaining_gumbels + sigma_values

        num_to_keep = max(1, len(remaining_actions) // 2)
        top_indices = np.argsort(scores)[-num_to_keep:]

        remaining_actions = remaining_actions[top_indices]
        remaining_gumbels = remaining_gumbels[top_indices]

    return int(remaining_actions[0])


def copy_env(env: BuchbergerEnv) -> BuchbergerEnv:
    """Create a copy of the environment state."""
    new_env = copy(env)
    new_env.generators = list(env.generators)
    new_env.pairs = list(env.pairs)
    return new_env


@eqx.filter_jit
def jit_inference(model: GumbelMuZeroModel, obs: tuple) -> tuple[Array, Array]:
    """JIT-compiled inference for a single observation."""
    return model(obs)


class GumbelMuZeroSearch:
    """
    Gumbel MuZero search algorithm.

    Implements the Gumbel MuZero planning algorithm which uses Gumbel-Top-k
    sampling with Sequential Halving for efficient action selection.
    """

    def __init__(
        self,
        model: GumbelMuZeroModel,
        env: BuchbergerEnv,
        config: GumbelMuZeroConfig,
    ):
        """
        Initialize Gumbel MuZero search.

        Args:
            model: Neural network model for policy and value.
            env: Environment template.
            config: Search configuration.
        """
        self.model = model
        self.env = env
        self.config = config

    def search(
        self,
        env: BuchbergerEnv,
        key: PRNGKeyArray,
    ) -> tuple[np.ndarray, float]:
        """
        Run Gumbel MuZero search from current state.

        Args:
            env: Current environment state.
            key: JAX random key.

        Returns:
            Tuple of (policy, value):
                - policy: Action probabilities (improved policy)
                - value: Root value estimate
        """
        num_polys = len(env.generators)
        obs = make_obs(env.generators, env.pairs)

        policy_logits, value = jit_inference(self.model, obs)
        policy_logits = np.array(policy_logits)
        value = float(value)

        valid_actions = self._get_valid_actions(env)

        if len(valid_actions) == 0:
            return np.zeros(num_polys * num_polys), value

        root = GumbelNode(env=copy_env(env))

        priors = self._compute_priors(policy_logits, valid_actions)
        for action, prior in priors.items():
            if action not in root.children:
                root.children[action] = GumbelNode(prior=prior)
            else:
                root.children[action].prior = prior

        k = min(self.config.max_num_considered_actions, len(valid_actions))
        actions, gumbel_values = gumbel_top_k(key, policy_logits, valid_actions, k)

        selected_action = sequential_halving(
            actions,
            gumbel_values,
            root,
            self.env,
            self.model,
            self.config,
        )

        policy = np.zeros(num_polys * num_polys, dtype=np.float32)
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )

        if total_visits > 0:
            for action, child in root.children.items():
                if child.visit_count > 0:
                    policy[action] = child.visit_count / total_visits
        else:
            policy[selected_action] = 1.0

        return policy, value

    def _get_valid_actions(self, env: BuchbergerEnv) -> list[int]:
        """Get list of valid actions (flattened pair indices)."""
        pairs = env.pairs
        num_polys = len(env.generators)
        return [i * num_polys + j for i, j in pairs]

    def _compute_priors(
        self, policy_logits: np.ndarray, valid_actions: list[int]
    ) -> dict[int, float]:
        """Compute prior probabilities over valid actions."""
        if not valid_actions:
            return {}

        valid_logits = policy_logits[valid_actions]
        max_logit = valid_logits.max()
        exp_logits = np.exp(valid_logits - max_logit)
        probs = exp_logits / exp_logits.sum()

        return {action: float(prob) for action, prob in zip(valid_actions, probs)}


@dataclass
class Experience:
    """A single experience from self-play."""

    observation: tuple
    policy: np.ndarray
    value: float
    num_polys: int


class ReplayBuffer:
    """Replay buffer for storing self-play experiences."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: list[Experience] = []
        self.position = 0

    def add(self, experiences: list[Experience]) -> None:
        """Add experiences to the buffer."""
        for exp in experiences:
            if len(self.buffer) < self.max_size:
                self.buffer.append(exp)
            else:
                self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences."""
        indices = np.random.choice(
            len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False
        )
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


def batch_experiences(
    experiences: list[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Batch a list of experiences for training."""
    batch_size = len(experiences)

    max_polys = max(exp.num_polys for exp in experiences)
    max_monoms = max(
        max(len(p) for p in exp.observation[0]) for exp in experiences
    )
    num_vars = len(experiences[0].observation[0][0][0])

    batched_ideals = np.zeros(
        (batch_size, max_polys, max_monoms, num_vars), dtype=np.float32
    )
    batched_monomial_masks = np.zeros(
        (batch_size, max_polys, max_monoms), dtype=bool
    )
    batched_poly_masks = np.zeros((batch_size, max_polys), dtype=bool)
    batched_selectables = np.full(
        (batch_size, max_polys, max_polys), -np.inf, dtype=np.float32
    )

    batched_policies = np.zeros((batch_size, max_polys * max_polys), dtype=np.float32)
    batched_values = np.zeros(batch_size, dtype=np.float32)
    loss_mask = np.ones(batch_size, dtype=np.float32)

    for i, exp in enumerate(experiences):
        ideal, selectables = exp.observation
        num_polys = len(ideal)

        batched_poly_masks[i, :num_polys] = True

        for j, poly in enumerate(ideal):
            p_len = len(poly)
            batched_ideals[i, j, :p_len] = poly
            batched_monomial_masks[i, j, :p_len] = True

        if selectables:
            rows, cols = zip(*selectables)
            batched_selectables[i, rows, cols] = 0.0

        original_policy = exp.policy
        original_num_polys = exp.num_polys

        for orig_action in range(len(original_policy)):
            if original_policy[orig_action] > 0:
                orig_i = orig_action // original_num_polys
                orig_j = orig_action % original_num_polys
                new_action = orig_i * max_polys + orig_j
                batched_policies[i, new_action] = original_policy[orig_action]

        batched_values[i] = exp.value

    batched_obs = {
        "ideals": batched_ideals,
        "monomial_masks": batched_monomial_masks,
        "poly_masks": batched_poly_masks,
        "selectables": batched_selectables,
    }

    return batched_obs, batched_policies, batched_values, loss_mask


def run_self_play_episode(
    model: GumbelMuZeroModel,
    env: BuchbergerEnv,
    config: GumbelMuZeroConfig,
    key: PRNGKeyArray,
) -> list[Experience]:
    """Run a single self-play episode."""
    env.reset()
    search = GumbelMuZeroSearch(model, env, config)

    experiences = []
    rewards = []
    done = False

    while not done:
        current_obs = make_obs(env.generators, env.pairs)
        num_polys = len(env.generators)

        key, subkey = jax.random.split(key)
        policy, _ = search.search(env, subkey)

        exp = Experience(
            observation=current_obs,
            policy=policy,
            value=0.0,
            num_polys=num_polys,
        )
        experiences.append(exp)

        policy_sum = policy.sum()
        if policy_sum > 0:
            normalized_policy = policy / policy_sum
        else:
            valid_actions = search._get_valid_actions(env)
            normalized_policy = np.zeros_like(policy)
            for a in valid_actions:
                normalized_policy[a] = 1.0 / len(valid_actions)

        action = int(np.random.choice(len(policy), p=normalized_policy))
        i, j = action // num_polys, action % num_polys

        _, reward, terminated, truncated, _ = env.step((i, j))
        rewards.append(reward)
        done = terminated or truncated

    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + config.gamma * G
        returns.insert(0, G)

    for exp, ret in zip(experiences, returns):
        exp.value = ret

    return experiences


def generate_self_play_data(
    model: GumbelMuZeroModel,
    env: BuchbergerEnv,
    num_episodes: int,
    config: GumbelMuZeroConfig,
    key: PRNGKeyArray,
) -> list[Experience]:
    """Generate self-play data from multiple episodes."""
    all_experiences = []

    for episode in tqdm(range(num_episodes), desc="Self-play"):
        key, subkey = jax.random.split(key)
        experiences = run_self_play_episode(model, env, config, subkey)
        all_experiences.extend(experiences)

    return all_experiences


def gumbel_muzero_loss(
    model: GumbelMuZeroModel,
    observations: dict,
    target_policies: Array,
    values: Array,
    loss_mask: Array,
) -> tuple[Array, dict]:
    """Compute combined policy and value loss."""
    policy_logits, pred_values = eqx.filter_vmap(model)(observations)

    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    policy_cross_entropy = jnp.where(
        target_policies > 0,
        -target_policies * log_probs,
        0.0,
    )
    policy_loss = jnp.sum(policy_cross_entropy, axis=-1)

    value_loss = (pred_values - values) ** 2

    total_loss = policy_loss + value_loss

    masked_loss = (total_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    masked_policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    masked_value_loss = (value_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)

    metrics = {
        "policy_loss": masked_policy_loss,
        "value_loss": masked_value_loss,
        "total_loss": masked_loss,
    }

    return masked_loss, metrics


def train_gumbel_muzero(
    model: GumbelMuZeroModel,
    replay_buffer: ReplayBuffer,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[GumbelMuZeroModel, optax.OptState, dict]:
    """Train the model on replay buffer data."""

    @eqx.filter_jit
    def make_step(
        model: GumbelMuZeroModel,
        opt_state: optax.OptState,
        observations: dict,
        target_policies: Array,
        values: Array,
        loss_mask: Array,
    ) -> tuple[GumbelMuZeroModel, optax.OptState, Array, dict]:
        def loss_fn(m):
            return gumbel_muzero_loss(m, observations, target_policies, values, loss_mask)

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics

    epoch_metrics = {"policy_loss": [], "value_loss": [], "total_loss": []}
    num_batches = max(1, len(replay_buffer) // train_config.batch_size)

    for _ in range(train_config.num_epochs_per_iteration):
        for _ in range(num_batches):
            batch = replay_buffer.sample(train_config.batch_size)
            observations, target_policies, values, loss_mask = batch_experiences(batch)

            observations = {k: jnp.array(v) for k, v in observations.items()}
            target_policies = jnp.array(target_policies)
            values = jnp.array(values)
            loss_mask = jnp.array(loss_mask)

            model, opt_state, _, metrics = make_step(
                model, opt_state, observations, target_policies, values, loss_mask
            )

            for k, v in metrics.items():
                epoch_metrics[k].append(float(v))

    mean_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
    return model, opt_state, mean_metrics


def evaluate_model(
    model: GumbelMuZeroModel,
    env: BuchbergerEnv,
    num_episodes: int = 20,
) -> dict:
    """Evaluate the model by playing episodes greedily."""
    episode_rewards = []
    episode_lengths = []

    for seed in range(num_episodes):
        env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            obs = make_obs(env.generators, env.pairs)
            policy_logits, _ = model(obs)
            policy_logits = np.array(policy_logits)

            valid_actions = []
            num_polys = len(env.generators)
            for i, j in env.pairs:
                valid_actions.append(i * num_polys + j)

            mask = np.full(policy_logits.shape, float("-inf"))
            for a in valid_actions:
                mask[a] = 0.0
            masked_logits = policy_logits + mask

            action = int(np.argmax(masked_logits))
            i, j = action // num_polys, action % num_polys

            _, reward, terminated, truncated, _ = env.step((i, j))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


def gumbel_muzero_training_loop(
    model: GumbelMuZeroModel,
    env: BuchbergerEnv,
    num_iterations: int,
    episodes_per_iteration: int,
    gumbel_config: GumbelMuZeroConfig,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    replay_buffer: ReplayBuffer,
    key: PRNGKeyArray,
    checkpoint_dir: str | None = None,
    eval_interval: int = 5,
    eval_episodes: int = 20,
) -> GumbelMuZeroModel:
    """
    Main Gumbel MuZero training loop.

    Alternates between self-play data generation and training.

    Args:
        model: Initial model.
        env: Environment for self-play.
        num_iterations: Number of training iterations.
        episodes_per_iteration: Self-play episodes per iteration.
        gumbel_config: Gumbel MuZero configuration.
        train_config: Training configuration.
        optimizer: Optax optimizer.
        replay_buffer: Replay buffer for storing experiences.
        key: JAX random key.
        checkpoint_dir: Directory for saving checkpoints.
        eval_interval: Evaluate every N iterations.
        eval_episodes: Number of episodes for evaluation.

    Returns:
        Trained model.
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    best_reward = float("-inf")

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        print("\nGenerating self-play data...")
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, episodes_per_iteration, gumbel_config, subkey
        )
        print(f"Generated {len(experiences)} experiences")

        replay_buffer.add(experiences)
        print(f"Replay buffer size: {len(replay_buffer)}")

        metrics: dict = {}
        if len(replay_buffer) >= train_config.batch_size:
            print("\nTraining...")
            model, opt_state, metrics = train_gumbel_muzero(
                model, replay_buffer, train_config, optimizer, opt_state
            )
            print(
                f"  Policy loss: {metrics['policy_loss']:.4f}, "
                f"Value loss: {metrics['value_loss']:.4f}, "
                f"Total loss: {metrics['total_loss']:.4f}"
            )

            if checkpoint_dir:
                save_checkpoint(
                    model, opt_state, checkpoint_dir, "last", iteration + 1, metrics
                )

        if (iteration + 1) % eval_interval == 0:
            print("\nEvaluating...")
            eval_metrics = evaluate_model(model, env, eval_episodes)
            print(
                f"  Mean reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}, "
                f"Mean length: {eval_metrics['mean_length']:.1f}"
            )

            if eval_metrics["mean_reward"] > best_reward:
                best_reward = eval_metrics["mean_reward"]
                if checkpoint_dir:
                    combined_metrics = {**metrics, **eval_metrics}
                    save_checkpoint(
                        model, opt_state, checkpoint_dir, "best", iteration + 1, combined_metrics
                    )
                    print(f"  Saved new best model (reward: {best_reward:.2f})")

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    return model


if __name__ == "__main__":
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)

    pretrained_checkpoint_path: str | None = "models/checkpoints/best.eqx"

    model_config = ModelConfig(
        monomials_dim=num_vars + 1,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=4,
        ideal_num_heads=8,
        value_hidden_dim=128,
    )

    gumbel_config = GumbelMuZeroConfig(
        num_simulations=16,
        max_num_considered_actions=16,
        gamma=0.99,
        c_visit=50.0,
        c_scale=1.0,
    )

    train_config = TrainConfig(
        learning_rate=1e-4,
        batch_size=32,
        num_epochs_per_iteration=3,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    )

    num_iterations = 50
    episodes_per_iteration = 5
    replay_buffer_size = 50000
    checkpoint_dir = os.path.join("models", "gumbel_muzero_checkpoints")
    eval_interval = 5
    eval_episodes = 10

    key = jax.random.key(42)

    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_gen, mode="train")

    optimizer = optax.nadam(train_config.learning_rate)

    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained model from {pretrained_checkpoint_path}")
        key, k_model = jax.random.split(key)
        model = GumbelMuZeroModel.from_pretrained(
            checkpoint_path=pretrained_checkpoint_path,
            config=model_config,
            optimizer=optimizer,
            key=k_model,
        )
        print("Pretrained policy loaded. Fresh value head initialized.")
    else:
        print("Initializing model from scratch")
        key, k_model = jax.random.split(key)
        model = GumbelMuZeroModel.from_scratch(config=model_config, key=k_model)

    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    print("\nStarting Gumbel MuZero training...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Episodes per iteration: {episodes_per_iteration}")
    print(f"  Simulations: {gumbel_config.num_simulations}")
    print(f"  Max considered actions: {gumbel_config.max_num_considered_actions}")
    print(f"  Replay buffer size: {replay_buffer_size}")
    print(f"  Checkpoint directory: {checkpoint_dir}")

    trained_model = gumbel_muzero_training_loop(
        model=model,
        env=env,
        num_iterations=num_iterations,
        episodes_per_iteration=episodes_per_iteration,
        gumbel_config=gumbel_config,
        train_config=train_config,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        key=key,
        checkpoint_dir=checkpoint_dir,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    print("\nDone!")
