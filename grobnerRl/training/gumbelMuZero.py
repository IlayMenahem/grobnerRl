"""
Gumbel MuZero implementation for the Buchberger environment.

This module implements the Gumbel MuZero algorithm which uses Gumbel-Top-k
sampling with Sequential Halving for efficient action selection without
requiring full tree search.

Key components aligned with the paper:
    - Gumbel-Top-k sampling: Sample m actions without replacement (Equations 3-5)
    - Sequential Halving: Iteratively halve action set, scoring by g(a) + logits(a) + σ(q̂(a))
    - Sigma transformation: σ(q̂(a)) = (c_visit + max_b N(b)) * c_scale * q̂(a) (Equation 8)
    - Completed Q-values: completedQ(a) = q(a) if visited, else v_π (Equation 10)
    - Improved policy: π' = softmax(logits + σ(completedQ)) (Equation 11)
    - Policy training: Cross-entropy with improved policy target (equivalent to KL divergence in Equation 12)

References:
    - "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)
"""

from copy import copy
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue
from grobnerRl.training.shared import Experience


@dataclass
class GumbelMuZeroConfig:
    """Configuration for Gumbel MuZero search."""

    num_simulations: int = 16
    max_num_considered_actions: int = 16
    gamma: float = 0.99
    c_visit: float = 50.0
    c_scale: float = 1.0


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


def sigma(
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """
    Compute the sigma transformation of Q-values as per Equation 8 in the paper.
    
    σ(q̂(a)) = (c_visit + max_b N(b)) * c_scale * q̂(a)

    Args:
        q_values: Q-values for each action.
        visit_counts: Visit counts for each action.
        c_visit: Visit count scaling constant.
        c_scale: Q-value scaling constant.

    Returns:
        Transformed Q-values (sigma values).
    """
    max_visit = visit_counts.max() if len(visit_counts) > 0 else 0
    scale_factor = (c_visit + max_visit) * c_scale
    return scale_factor * q_values


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


def copy_env(env: BuchbergerEnv) -> BuchbergerEnv:
    """Create a copy of the environment state."""
    new_env = copy(env)
    new_env.generators = list(env.generators)
    new_env.pairs = list(env.pairs)
    return new_env


def sequential_halving(
    actions: np.ndarray,
    gumbel_values: np.ndarray,
    policy_logits: np.ndarray,
    root: GumbelNode,
    env_template: BuchbergerEnv,
    model: GrobnerPolicyValue,
    config: GumbelMuZeroConfig,
) -> int:
    """
    Sequential Halving procedure for action selection.

    Iteratively reduces the action set by half, allocating simulations
    to remaining actions and eliminating the worst half.

    Args:
        actions: Initial set of candidate actions.
        gumbel_values: Gumbel values for each action.
        policy_logits: Full policy logits from neural network.
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
                # Expand node if not yet expanded
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
                    # Backup the bootstrapped value for newly expanded node
                    child = root.children[action]
                    child.visit_count += 1
                    bootstrapped_value = reward + config.gamma * child_value
                    child.value_sum += bootstrapped_value
                else:
                    child = root.children[action]
                    child.visit_count += 1
                    if child.is_terminal:
                        bootstrapped_value = child.reward
                    else:
                        obs = make_obs(child.env.generators, child.env.pairs)
                        _, child_value = model(obs)
                        bootstrapped_value = child.reward + config.gamma * float(child_value)
                    child.value_sum += bootstrapped_value

        q_values = np.array(
            [
                root.children[int(a)].q_value if int(a) in root.children else 0.0
                for a in remaining_actions
            ]
        )
        visit_counts = np.array(
            [
                root.children[int(a)].visit_count if int(a) in root.children else 0
                for a in remaining_actions
            ]
        )

        action_logits = policy_logits[remaining_actions]
        sigma_values = sigma(
            q_values, visit_counts, config.c_visit, config.c_scale
        )
        scores = remaining_gumbels + action_logits + sigma_values

        num_to_keep = max(1, len(remaining_actions) // 2)
        top_indices = np.argsort(scores)[-num_to_keep:]

        remaining_actions = remaining_actions[top_indices]
        remaining_gumbels = remaining_gumbels[top_indices]

    return int(remaining_actions[0])


@eqx.filter_jit
def jit_inference(model: GrobnerPolicyValue, obs: tuple) -> tuple[Array, Array]:
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
        model: GrobnerPolicyValue,
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
            policy_logits,
            root,
            self.env,
            self.model,
            self.config,
        )

        # Construct improved policy using completed Q-values (Equations 10-11)
        # completedQ(a) = q(a) if N(a) > 0 else v_π
        
        # Compute v_π as weighted average of visited Q-values by priors
        v_pi = 0.0
        prior_sum = 0.0
        for action in root.children:
            if root.children[action].visit_count > 0:
                v_pi += root.children[action].prior * root.children[action].q_value
                prior_sum += root.children[action].prior
        
        if prior_sum > 0:
            v_pi /= prior_sum
        else:
            v_pi = value  # Fall back to network value
        
        # Build completed Q-values
        completed_q = np.full(num_polys * num_polys, v_pi, dtype=np.float32)
        for action, child in root.children.items():
            if child.visit_count > 0:
                completed_q[action] = child.q_value
        
        # Compute sigma transformation for all actions
        visit_counts = np.zeros(num_polys * num_polys)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        sigma_values = sigma(
            completed_q, visit_counts, self.config.c_visit, self.config.c_scale
        )
        
        # Construct improved policy: π' = softmax(logits + σ(completedQ))
        improved_logits = policy_logits + sigma_values
        
        # Mask invalid actions
        mask = np.full_like(improved_logits, -np.inf)
        for action in valid_actions:
            mask[action] = 0.0
        improved_logits = improved_logits + mask
        
        # Softmax to get improved policy
        max_logit = improved_logits.max()
        exp_logits = np.exp(improved_logits - max_logit)
        policy = exp_logits / exp_logits.sum()

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


def run_self_play_episode(
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    config: GumbelMuZeroConfig,
    key: PRNGKeyArray,
    poly_cache,
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

        exp = Experience.from_uncompressed(
            observation=current_obs,
            policy=policy,
            value=0.0,
            num_polys=num_polys,
            poly_cache=poly_cache,
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
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    num_episodes: int,
    config: GumbelMuZeroConfig,
    key: PRNGKeyArray,
    poly_cache,
) -> list[Experience]:
    """Generate self-play data from multiple episodes."""
    from tqdm import tqdm

    all_experiences = []

    for episode in tqdm(range(num_episodes), desc="Self-play"):
        key, subkey = jax.random.split(key)
        experiences = run_self_play_episode(model, env, config, subkey, poly_cache)
        all_experiences.extend(experiences)

    return all_experiences
