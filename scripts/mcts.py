"""
AlphaZero-style MCTS training for Buchberger environment.

This script implements:
- MCTS with neural network guidance (policy + value)
- Self-play data generation
- Combined policy and value training
- Support for pretrained models from supervised_jax.py
"""

import os
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module
from jaxtyping import Array
from tqdm import tqdm

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.models import (
    Extractor,
    GrobnerPolicy,
    IdealModel,
    MonomialEmbedder,
    PairwiseScorer,
    PolynomialEmbedder,
)
from grobnerRl.types import Observation


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================


def load_checkpoint(checkpoint_path: str, template: dict) -> dict:
    """
    Load a checkpoint saved by supervised_jax.py save_checkpoint().

    Args:
        checkpoint_path: Path to the .eqx checkpoint file.
        template: A dictionary with the same structure as the saved payload,
                  containing model templates for deserialization.

    Returns:
        Dictionary containing: model, opt_state, epoch, val_accuracy
    """
    with open(checkpoint_path, "rb") as f:
        payload = eqx.tree_deserialise_leaves(f, template)
    return payload


def load_pretrained_policy(
    checkpoint_path: str,
    monomials_dim: int,
    monoms_embedding_dim: int,
    polys_embedding_dim: int,
    ideal_depth: int,
    ideal_num_heads: int,
    optimizer: optax.GradientTransformation,
    key: Array,
) -> GrobnerPolicy:
    """
    Load a pretrained GrobnerPolicy from a supervised training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        monomials_dim: Dimension of monomial features (num_vars + 1).
        monoms_embedding_dim: Embedding dimension for monomials.
        polys_embedding_dim: Embedding dimension for polynomials.
        ideal_depth: Number of transformer layers in IdealModel.
        ideal_num_heads: Number of attention heads in IdealModel.
        optimizer: Optimizer to create template opt_state.
        key: JAX random key for creating template model.

    Returns:
        Loaded GrobnerPolicy with pretrained weights.
    """
    # Create template model with same architecture
    k_monomial, k_polynomial, k_ideal, k_scorer = jax.random.split(key, 4)

    monomial_embedder = MonomialEmbedder(monomials_dim, monoms_embedding_dim, k_monomial)
    polynomial_embedder = PolynomialEmbedder(
        input_dim=monoms_embedding_dim,
        hidden_dim=polys_embedding_dim,
        hidden_layers=2,
        output_dim=polys_embedding_dim,
        key=k_polynomial,
    )
    ideal_model = IdealModel(polys_embedding_dim, ideal_num_heads, ideal_depth, k_ideal)
    pairwise_scorer = PairwiseScorer(polys_embedding_dim, polys_embedding_dim, k_scorer)
    extractor = Extractor(
        monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer
    )
    template_policy = GrobnerPolicy(extractor)

    # Create template optimizer state
    template_opt_state = optimizer.init(eqx.filter(template_policy, eqx.is_array))

    # Create full template for deserialization
    template = {
        "model": template_policy,
        "opt_state": template_opt_state,
        "epoch": 0,
        "val_accuracy": 0.0,
    }

    payload = load_checkpoint(checkpoint_path, template)
    return payload["model"]


def save_checkpoint(
    model: Module,
    opt_state: optax.OptState,
    checkpoint_dir: str,
    label: str,
    iteration: int,
    metrics: dict,
) -> str:
    """
    Save a checkpoint during AlphaZero training.

    Args:
        model: The GrobnerAlphaZero model to save.
        opt_state: Current optimizer state.
        checkpoint_dir: Directory to save checkpoints.
        label: Label for the checkpoint (e.g., 'last', 'best').
        iteration: Current training iteration.
        metrics: Dictionary of training metrics.

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"alphazero_{label}.eqx")
    payload = {
        "model": model,
        "opt_state": opt_state,
        "iteration": iteration,
        "metrics": metrics,
    }
    with open(ckpt_path, "wb") as f:
        eqx.tree_serialise_leaves(f, payload)
    return ckpt_path


# =============================================================================
# AlphaZero Model
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    monomials_dim: int
    monoms_embedding_dim: int = 64
    polys_embedding_dim: int = 128
    ideal_depth: int = 4
    ideal_num_heads: int = 8
    value_hidden_dim: int = 128


class GrobnerAlphaZero(Module):
    """
    AlphaZero-style model with shared backbone for policy and value.

    The policy head outputs logits over all possible (i, j) pair actions.
    The value head outputs a scalar estimate of expected return.
    """

    extractor: Extractor
    value_head: eqx.nn.MLP

    def __init__(self, extractor: Extractor, value_head: eqx.nn.MLP):
        self.extractor = extractor
        self.value_head = value_head

    def __call__(self, obs: Observation | dict | tuple) -> tuple[Array, Array]:
        """
        Forward pass returning both policy logits and value estimate.

        Args:
            obs: Observation from the environment (tuple or dict format).

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Flattened logits over (i, j) pairs
                - value: Scalar value estimate
        """
        # Get policy logits from extractor (same as GrobnerPolicy)
        policy_logits = self.extractor(obs)

        # Get value from pooling ideal embeddings
        # We need to compute the intermediate embeddings
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

            # Mean pooling over valid polynomials for value
            masked_embs = jnp.where(poly_mask[:, None], ideal_embeddings, 0.0)
            pooled = masked_embs.sum(axis=0) / (poly_mask.sum() + 1e-9)
        else:
            ideal, selectables = obs

            # Pad and embed (similar to Extractor)
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

            # Mean pooling for value
            pooled = ideal_embeddings.mean(axis=0)

        # Value head
        value = self.value_head(pooled).squeeze(-1)

        return policy_logits, value

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: ModelConfig,
        optimizer: optax.GradientTransformation,
        key: Array,
    ) -> "GrobnerAlphaZero":
        """
        Initialize from a pretrained GrobnerPolicy checkpoint.

        Loads the Extractor weights from supervised training and
        initializes a fresh value head.

        Args:
            checkpoint_path: Path to supervised training checkpoint.
            config: Model configuration.
            optimizer: Optimizer for creating template opt_state.
            key: JAX random key.

        Returns:
            GrobnerAlphaZero with pretrained policy weights.
        """
        key, k_value, k_load = jax.random.split(key, 3)

        # Load pretrained policy
        pretrained_policy = load_pretrained_policy(
            checkpoint_path=checkpoint_path,
            monomials_dim=config.monomials_dim,
            monoms_embedding_dim=config.monoms_embedding_dim,
            polys_embedding_dim=config.polys_embedding_dim,
            ideal_depth=config.ideal_depth,
            ideal_num_heads=config.ideal_num_heads,
            optimizer=optimizer,
            key=k_load,
        )

        # Extract the shared backbone
        extractor = pretrained_policy.extractor

        # Create fresh value head
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
    def from_scratch(
        cls,
        config: ModelConfig,
        key: Array,
    ) -> "GrobnerAlphaZero":
        """
        Initialize model from scratch (no pretraining).

        Args:
            config: Model configuration.
            key: JAX random key.

        Returns:
            Fresh GrobnerAlphaZero model.
        """
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


# =============================================================================
# MCTS Configuration and Node
# =============================================================================


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 100
    c_puct: float = 1.0
    gamma: float = 0.99
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    batch_size: int = 8
    use_batched_mcts: bool = True


@dataclass
class MCTSNode:
    """
    A node in the MCTS search tree.

    Attributes:
        state: Environment state at this node (generators, pairs, env copy).
        parent: Parent node (None for root).
        action: Action that led to this node from parent.
        children: Dictionary mapping actions to child nodes (created lazily).
        priors: Dictionary mapping actions to prior probabilities (for lazy expansion).
        valid_actions: List of valid actions at this node (for lazy expansion).
        visit_count: Number of times this node was visited.
        value_sum: Sum of values backed up through this node.
        prior: Prior probability from policy network (for this node from parent).
        is_terminal: Whether this is a terminal state.
        reward: Immediate reward received when reaching this node.
        virtual_loss: Virtual loss for parallel search (tracks in-flight visits).
    """

    state: dict = field(default_factory=dict)
    parent: "MCTSNode | None" = None
    action: int | None = None
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    priors: dict[int, float] = field(default_factory=dict)
    valid_actions: list[int] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    reward: float = 0.0
    virtual_loss: int = 0

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a), accounting for virtual loss."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Pessimistic estimate during parallel search
        return self.value_sum / total_visits

    def is_expanded(self) -> bool:
        """Check if this node has been expanded (priors computed)."""
        return len(self.priors) > 0 or self.is_terminal


# =============================================================================
# JIT-Compiled Inference Functions
# =============================================================================


@eqx.filter_jit
def _jit_single_inference(model: GrobnerAlphaZero, obs: tuple) -> tuple[Array, Array]:
    """JIT-compiled inference for a single observation."""
    return model(obs)


@eqx.filter_jit
def _jit_batched_inference(model: GrobnerAlphaZero, batched_obs: dict) -> tuple[Array, Array]:
    """JIT-compiled batched inference using filter_vmap."""
    return eqx.filter_vmap(model)(batched_obs)


# =============================================================================
# MCTS Search
# =============================================================================


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT (Predictor + Upper Confidence bounds for Trees) for selection,
    and a neural network for policy priors and value estimation.
    """

    def __init__(
        self,
        model: GrobnerAlphaZero,
        env: BuchbergerEnv,
        config: MCTSConfig,
    ):
        """
        Initialize MCTS.

        Args:
            model: AlphaZero model providing policy and value.
            env: Environment template for simulations.
            config: MCTS configuration.
        """
        self.model = model
        self.env = env
        self.config = config
        self._jit_inference = _jit_single_inference

    def _get_observation(self, env: BuchbergerEnv):
        """Get observation from environment state."""
        from grobnerRl.envs.env import make_obs

        return make_obs(env.generators, env.pairs)

    def _get_valid_actions(self, env: BuchbergerEnv) -> list[int]:
        """Get list of valid actions (flattened pair indices)."""
        pairs = env.pairs
        num_polys = len(env.generators)
        valid_actions = []
        for i, j in pairs:
            action = i * num_polys + j
            valid_actions.append(action)
        return valid_actions

    def _copy_env(self, env: BuchbergerEnv) -> BuchbergerEnv:
        """Create a copy of the environment state.
        
        We avoid deepcopy because SymPy's PolyRing objects don't copy properly.
        Instead, we create a new env and manually copy the state.
        """
        from copy import copy
        
        new_env = copy(env)
        # Shallow copy the lists (polynomials themselves are immutable-ish in usage)
        new_env.generators = list(env.generators)
        new_env.pairs = list(env.pairs)
        return new_env

    def _copy_env_with_obs(self, env: BuchbergerEnv) -> tuple[BuchbergerEnv, tuple]:
        """Create a copy of the environment state with cached observation.
        
        Args:
            env: Environment to copy.
            
        Returns:
            Tuple of (new_env, cached_observation).
        """
        new_env = self._copy_env(env)
        obs = self._get_observation(new_env)
        return new_env, obs

    def _inference(self, obs: tuple) -> tuple[Array, Array]:
        """
        Run neural network inference (uses JIT-compiled function).

        Args:
            obs: Observation tuple (ideal, selectables).

        Returns:
            Tuple of (policy_logits, value).
        """
        policy_logits, value = self._jit_inference(self.model, obs)
        return policy_logits, value

    def search(
        self, root_env: BuchbergerEnv, add_noise: bool = True
    ) -> tuple[np.ndarray, float]:
        """
        Run MCTS from the current state.

        Args:
            root_env: Environment at the root state.
            add_noise: Whether to add Dirichlet noise to root priors.

        Returns:
            Tuple of (policy, value):
                - policy: Visit count distribution over actions (max_polys^2)
                - value: Root node value estimate
        """
        # Create root node
        root = MCTSNode(
            state={"env": self._copy_env(root_env)},
            parent=None,
            action=None,
        )

        # Expand root
        self._expand(root, add_noise=add_noise)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree until we find an unexpanded node
            while node.is_expanded() and not node.is_terminal:
                node = self._select(node)
                search_path.append(node)

            # Expansion and evaluation
            if not node.is_terminal:
                value = self._expand(node, add_noise=False)
            else:
                # Terminal node: value is 0 (no more rewards)
                value = 0.0

            # Backup
            self._backup(search_path, value)

        # Compute policy from visit counts (only over valid actions)
        num_polys = len(root_env.generators)
        policy = np.zeros(num_polys * num_polys, dtype=np.float32)

        if not root.valid_actions:
            return policy, root.q_value

        if self.config.temperature == 0:
            if root.children:
                best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
                policy[best_action] = 1.0
        else:
            visit_counts = np.array([
                root.children[a].visit_count if a in root.children else 0
                for a in root.valid_actions
            ], dtype=np.float32)

            if visit_counts.sum() > 0:
                visit_counts_temp = visit_counts ** (1.0 / self.config.temperature)
                probs = visit_counts_temp / visit_counts_temp.sum()
                # Map back to full policy array
                for i, action in enumerate(root.valid_actions):
                    policy[action] = probs[i]

        return policy, root.q_value

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using PUCT formula (vectorized, with lazy child creation).

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            node: Current node to select from.

        Returns:
            Selected child node.
        """
        if not node.valid_actions:
            raise ValueError("No valid actions - node should be expanded first")

        # Vectorized PUCT computation
        actions = np.array(node.valid_actions)
        priors = np.array([node.priors[a] for a in actions])

        visits = np.array([
            node.children[a].visit_count + node.children[a].virtual_loss
            if a in node.children else 0
            for a in actions
        ])
        q_values = np.array([
            node.children[a].q_value if a in node.children else 0.0
            for a in actions
        ])

        sqrt_parent = np.sqrt(node.visit_count + 1)
        exploration = self.config.c_puct * priors * sqrt_parent / (1 + visits)
        scores = q_values + exploration

        best_idx = int(np.argmax(scores))
        best_action = actions[best_idx]

        return self._get_or_create_child(node, best_action)

    def _expand(self, node: MCTSNode, add_noise: bool = False) -> float:
        """
        Expand a node by computing priors (lazy expansion - children created on demand).

        Args:
            node: Node to expand.
            add_noise: Whether to add Dirichlet noise to priors.

        Returns:
            Value estimate from the neural network.
        """
        env = node.state["env"]

        # Check if terminal
        if len(env.pairs) == 0:
            node.is_terminal = True
            return 0.0

        # Use cached observation if available, otherwise compute
        obs = node.state.get("cached_obs")
        if obs is None:
            obs = self._get_observation(env)
            node.state["cached_obs"] = obs  # Cache for potential reuse
        
        policy_logits, value = self._inference(obs)

        # Convert to numpy
        policy_logits = np.array(policy_logits)
        value = float(value)

        # Get valid actions
        valid_actions = self._get_valid_actions(env)
        node.valid_actions = valid_actions

        # Compute priors only over valid actions (efficient masking)
        priors_dict = self._compute_priors(policy_logits, valid_actions, add_noise)
        node.priors = priors_dict

        return value

    def _compute_priors(
        self, policy_logits: np.ndarray, valid_actions: list[int], add_noise: bool
    ) -> dict[int, float]:
        """
        Compute prior probabilities only over valid actions.

        Args:
            policy_logits: Raw logits from the neural network.
            valid_actions: List of valid action indices.
            add_noise: Whether to add Dirichlet noise.

        Returns:
            Dictionary mapping action -> prior probability.
        """
        if not valid_actions:
            return {}

        # Extract logits only for valid actions
        valid_logits = policy_logits[valid_actions]

        # Softmax over valid actions only
        max_logit = valid_logits.max()
        exp_logits = np.exp(valid_logits - max_logit)
        probs = exp_logits / exp_logits.sum()

        # Add Dirichlet noise if requested
        if add_noise:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(valid_actions)
            )
            probs = (1 - self.config.dirichlet_epsilon) * probs + self.config.dirichlet_epsilon * noise

        # Return as dictionary
        return {action: float(prob) for action, prob in zip(valid_actions, probs)}

    def _get_or_create_child(self, parent: MCTSNode, action: int) -> MCTSNode:
        """
        Get existing child or create it lazily with cached observation.

        Args:
            parent: Parent node.
            action: Action to get/create child for.

        Returns:
            Child node for the given action.
        """
        if action in parent.children:
            return parent.children[action]

        # Create child lazily
        env = parent.state["env"]
        child_env = self._copy_env(env)
        num_polys = len(env.generators)

        # Convert flat action to pair and step
        i, j = action // num_polys, action % num_polys
        _, reward, terminated, _, _ = child_env.step((i, j))

        # Pre-compute and cache observation for this child
        cached_obs = self._get_observation(child_env) if not terminated else None

        child = MCTSNode(
            state={"env": child_env, "cached_obs": cached_obs},
            parent=parent,
            action=action,
            prior=parent.priors.get(action, 0.0),
            is_terminal=terminated,
            reward=reward,
        )
        parent.children[action] = child
        return child

    def _backup(self, search_path: list[MCTSNode], value: float) -> None:
        """
        Backup value through the search path.

        Args:
            search_path: List of nodes from root to leaf.
            value: Value estimate at the leaf.
        """
        # Traverse path from leaf to root
        for node in reversed(search_path):
            node.visit_count += 1
            # Remove virtual loss added during selection
            node.virtual_loss = max(0, node.virtual_loss - 1)
            # Include reward when backing up
            value = node.reward + self.config.gamma * value
            node.value_sum += value

    def _add_virtual_loss(self, search_path: list[MCTSNode]) -> None:
        """Add virtual loss to nodes in search path for parallel search."""
        for node in search_path:
            node.virtual_loss += 1

    def _select_leaf(self, root: MCTSNode) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Select a leaf node from the tree (for batched search).

        Applies virtual loss during traversal to encourage exploration
        of different paths in parallel.

        Args:
            root: Root node to start from.

        Returns:
            Tuple of (leaf_node, search_path).
        """
        node = root
        search_path = [node]

        while node.is_expanded() and not node.is_terminal:
            node = self._select(node)
            search_path.append(node)

        # Add virtual loss to encourage different paths
        self._add_virtual_loss(search_path)

        return node, search_path

    def _batch_observations(
        self, observations: list[tuple]
    ) -> dict:
        """
        Batch multiple observations into a single dict for batched inference.

        Args:
            observations: List of (ideal, selectables) tuples.

        Returns:
            Batched observation dict ready for filter_vmap.
        """
        batch_size = len(observations)
        if batch_size == 0:
            return {}

        # Calculate dimensions
        max_polys = max(len(obs[0]) for obs in observations)
        max_monoms = max(
            max(len(p) for p in obs[0]) if obs[0] else 1
            for obs in observations
        )
        num_vars = len(observations[0][0][0][0]) if observations[0][0] else 6

        # Allocate buffers
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

        for i, (ideal, selectables) in enumerate(observations):
            num_polys = len(ideal)
            batched_poly_masks[i, :num_polys] = True

            for j, poly in enumerate(ideal):
                p_len = len(poly)
                batched_ideals[i, j, :p_len] = poly
                batched_monomial_masks[i, j, :p_len] = True

            if selectables:
                rows, cols = zip(*selectables)
                batched_selectables[i, rows, cols] = 0.0

        return {
            "ideals": jnp.array(batched_ideals),
            "monomial_masks": jnp.array(batched_monomial_masks),
            "poly_masks": jnp.array(batched_poly_masks),
            "selectables": jnp.array(batched_selectables),
        }

    def _batched_inference(self, batched_obs: dict) -> tuple[Array, Array]:
        """
        Run batched neural network inference.

        Args:
            batched_obs: Batched observation dict.

        Returns:
            Tuple of (policy_logits_batch, values_batch).
        """
        return _jit_batched_inference(self.model, batched_obs)

    def _expand_with_result(
        self,
        node: MCTSNode,
        policy_logits: np.ndarray,
        value: float,
        add_noise: bool = False,
    ) -> float:
        """
        Expand a node using pre-computed neural network results.

        Args:
            node: Node to expand.
            policy_logits: Pre-computed policy logits.
            value: Pre-computed value estimate.
            add_noise: Whether to add Dirichlet noise.

        Returns:
            Value estimate.
        """
        env = node.state["env"]

        if len(env.pairs) == 0:
            node.is_terminal = True
            return 0.0

        valid_actions = self._get_valid_actions(env)
        node.valid_actions = valid_actions

        priors_dict = self._compute_priors(policy_logits, valid_actions, add_noise)
        node.priors = priors_dict

        return value

    def search_batched(
        self, root_env: BuchbergerEnv, add_noise: bool = True, batch_size: int = 8
    ) -> tuple[np.ndarray, float]:
        """
        Run MCTS with batched neural network inference.

        Collects multiple leaves and evaluates them in a single batched
        forward pass for improved efficiency.

        Args:
            root_env: Environment at the root state.
            add_noise: Whether to add Dirichlet noise to root priors.
            batch_size: Number of leaves to batch together.

        Returns:
            Tuple of (policy, value):
                - policy: Visit count distribution over actions
                - value: Root node value estimate
        """
        # Create root node
        root = MCTSNode(
            state={"env": self._copy_env(root_env)},
            parent=None,
            action=None,
        )

        # Expand root (single inference)
        self._expand(root, add_noise=add_noise)

        # Run simulations in batches
        remaining_sims = self.config.num_simulations
        while remaining_sims > 0:
            current_batch_size = min(batch_size, remaining_sims)

            # Collect leaves
            leaves_and_paths: list[tuple[MCTSNode, list[MCTSNode]]] = []
            terminal_paths: list[list[MCTSNode]] = []

            for _ in range(current_batch_size):
                leaf, path = self._select_leaf(root)
                if leaf.is_terminal:
                    terminal_paths.append(path)
                elif not leaf.is_expanded():
                    leaves_and_paths.append((leaf, path))
                else:
                    # Already expanded (shouldn't happen often)
                    terminal_paths.append(path)

            # Backup terminal nodes immediately
            for path in terminal_paths:
                self._backup(path, 0.0)

            # Batch evaluate non-terminal leaves
            if leaves_and_paths:
                leaves, paths = zip(*leaves_and_paths)

                # Collect observations (use cached if available)
                observations = []
                for leaf in leaves:
                    obs = leaf.state.get("cached_obs")
                    if obs is None:
                        obs = self._get_observation(leaf.state["env"])
                        leaf.state["cached_obs"] = obs
                    observations.append(obs)

                # Batch inference
                batched_obs = self._batch_observations(observations)
                policy_logits_batch, values_batch = self._batched_inference(batched_obs)

                # Convert to numpy
                policy_logits_batch = np.array(policy_logits_batch)
                values_batch = np.array(values_batch)

                # Expand leaves and backup
                for i, (leaf, path) in enumerate(zip(leaves, paths)):
                    # Get the correct slice of policy logits for this leaf
                    num_polys = len(leaf.state["env"].generators)
                    max_polys = batched_obs["poly_masks"].shape[1]

                    # Reshape policy logits from flattened max_polys^2 to original num_polys^2
                    flat_logits = policy_logits_batch[i]
                    # Create remapped logits for original action space
                    remapped_logits = np.full(num_polys * num_polys, -np.inf)
                    for orig_action in range(num_polys * num_polys):
                        orig_i, orig_j = orig_action // num_polys, orig_action % num_polys
                        batched_action = orig_i * max_polys + orig_j
                        if batched_action < len(flat_logits):
                            remapped_logits[orig_action] = flat_logits[batched_action]

                    value = self._expand_with_result(
                        leaf, remapped_logits, float(values_batch[i]), add_noise=False
                    )
                    self._backup(path, value)

            remaining_sims -= current_batch_size

        # Compute policy from visit counts
        num_polys = len(root_env.generators)
        policy = np.zeros(num_polys * num_polys, dtype=np.float32)

        if not root.valid_actions:
            return policy, root.q_value

        if self.config.temperature == 0:
            if root.children:
                best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
                policy[best_action] = 1.0
        else:
            visit_counts = np.array([
                root.children[a].visit_count if a in root.children else 0
                for a in root.valid_actions
            ], dtype=np.float32)

            if visit_counts.sum() > 0:
                visit_counts_temp = visit_counts ** (1.0 / self.config.temperature)
                probs = visit_counts_temp / visit_counts_temp.sum()
                for i, action in enumerate(root.valid_actions):
                    policy[action] = probs[i]

        return policy, root.q_value


# =============================================================================
# Self-Play Data Generation
# =============================================================================


@dataclass
class Experience:
    """
    A single experience from self-play.

    Attributes:
        observation: Tokenized observation from the environment.
        mcts_policy: Visit count distribution from MCTS (target for policy).
        value: Discounted return from this state (target for value).
        num_polys: Number of polynomials at this state (for batching).
    """

    observation: tuple  # (ideal, selectables) from make_obs
    mcts_policy: np.ndarray
    value: float
    num_polys: int


class ReplayBuffer:
    """
    Replay buffer for storing self-play experiences.

    Stores experiences and provides batched sampling for training.
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of experiences to store.
        """
        self.max_size = max_size
        self.buffer: list[Experience] = []
        self.position = 0

    def add(self, experiences: list[Experience]) -> None:
        """
        Add experiences to the buffer.

        Args:
            experiences: List of experiences to add.
        """
        for exp in experiences:
            if len(self.buffer) < self.max_size:
                self.buffer.append(exp)
            else:
                self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled experiences.
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


def batch_experiences(
    experiences: list[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch a list of experiences for training.

    Args:
        experiences: List of Experience objects.

    Returns:
        Tuple of (observations, mcts_policies, values, loss_mask):
            - observations: Batched observation dict
            - mcts_policies: Batched MCTS policies
            - values: Batched value targets
            - loss_mask: Mask for valid samples
    """
    batch_size = len(experiences)

    # Calculate dimensions
    max_polys = max(exp.num_polys for exp in experiences)
    max_monoms = max(
        max(len(p) for p in exp.observation[0]) for exp in experiences
    )
    num_vars = len(experiences[0].observation[0][0][0])

    # Allocate buffers
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

        # Remap policy to max_polys grid
        original_policy = exp.mcts_policy
        original_num_polys = exp.num_polys

        for orig_action in range(len(original_policy)):
            if original_policy[orig_action] > 0:
                orig_i, orig_j = orig_action // original_num_polys, orig_action % original_num_polys
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
    model: GrobnerAlphaZero,
    env: BuchbergerEnv,
    mcts_config: MCTSConfig,
    seed: int | None = None,
) -> list[Experience]:
    """
    Run a single self-play episode using MCTS.

    Args:
        model: AlphaZero model for MCTS guidance.
        env: Environment to play in.
        mcts_config: MCTS configuration.
        seed: Random seed for environment reset.

    Returns:
        List of experiences collected during the episode.
    """
    from grobnerRl.envs.env import make_obs

    # Reset environment
    obs, _ = env.reset(seed=seed)

    # Create MCTS instance
    mcts = MCTS(model, env, mcts_config)

    experiences = []
    rewards = []
    done = False

    while not done:
        # Get current observation
        current_obs = make_obs(env.generators, env.pairs)
        num_polys = len(env.generators)

        # Run MCTS (use batched or standard based on config)
        if mcts_config.use_batched_mcts:
            policy, _ = mcts.search_batched(
                env, add_noise=True, batch_size=mcts_config.batch_size
            )
        else:
            policy, _ = mcts.search(env, add_noise=True)

        # Store experience (value will be computed later)
        exp = Experience(
            observation=current_obs,
            mcts_policy=policy,
            value=0.0,  # Placeholder
            num_polys=num_polys,
        )
        experiences.append(exp)

        # Sample action from policy
        if mcts_config.temperature == 0:
            action = int(np.argmax(policy))
        else:
            # Normalize policy to handle numerical issues
            policy_sum = policy.sum()
            if policy_sum > 0:
                normalized_policy = policy / policy_sum
            else:
                # Fallback to uniform over valid actions
                valid_actions = mcts._get_valid_actions(env)
                normalized_policy = np.zeros_like(policy)
                for a in valid_actions:
                    normalized_policy[a] = 1.0 / len(valid_actions)
            action = int(np.random.choice(len(policy), p=normalized_policy))

        # Convert to pair and step
        i, j = action // num_polys, action % num_polys
        _, reward, terminated, truncated, _ = env.step((i, j))

        rewards.append(reward)
        done = terminated or truncated

    # Compute discounted returns (value targets)
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + mcts_config.gamma * G
        returns.insert(0, G)

    # Update experience values
    for exp, ret in zip(experiences, returns):
        exp.value = ret

    return experiences


def generate_self_play_data(
    model: GrobnerAlphaZero,
    env: BuchbergerEnv,
    num_episodes: int,
    mcts_config: MCTSConfig,
) -> list[Experience]:
    """
    Generate self-play data from multiple episodes.

    Args:
        model: AlphaZero model for MCTS guidance.
        env: Environment template.
        num_episodes: Number of episodes to generate.
        mcts_config: MCTS configuration.

    Returns:
        List of all experiences from all episodes.
    """
    all_experiences = []

    for episode in tqdm(range(num_episodes), desc="Self-play"):
        experiences = run_self_play_episode(model, env, mcts_config, seed=episode)
        all_experiences.extend(experiences)

    return all_experiences


# =============================================================================
# Training Functions
# =============================================================================


@dataclass
class TrainConfig:
    """Configuration for AlphaZero training."""

    learning_rate: float = 1e-4
    batch_size: int = 128
    num_epochs_per_iteration: int = 10
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


def alphazero_loss(
    model: GrobnerAlphaZero,
    observations: dict,
    mcts_policies: Array,
    values: Array,
    loss_mask: Array,
) -> tuple[Array, dict]:
    """
    Compute combined policy and value loss for AlphaZero.

    Args:
        model: The GrobnerAlphaZero model.
        observations: Batched observations dict.
        mcts_policies: Target MCTS policies (batch_size, max_actions).
        values: Target values (batch_size,).
        loss_mask: Mask for valid samples (batch_size,).

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    # Forward pass
    policy_logits, pred_values = eqx.filter_vmap(model)(observations)

    # Policy loss: cross-entropy with MCTS policy
    # Use where to avoid 0 * -inf = nan when mcts_policies is 0
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    # Only compute loss where mcts_policies > 0, else use 0
    policy_cross_entropy = jnp.where(
        mcts_policies > 0,
        -mcts_policies * log_probs,
        0.0
    )
    policy_loss = jnp.sum(policy_cross_entropy, axis=-1)

    # Value loss: MSE
    value_loss = (pred_values - values) ** 2

    # Combine losses
    total_loss = policy_loss + value_loss

    # Apply mask and compute mean
    masked_loss = (total_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    masked_policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    masked_value_loss = (value_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)

    metrics = {
        "policy_loss": masked_policy_loss,
        "value_loss": masked_value_loss,
        "total_loss": masked_loss,
    }

    return masked_loss, metrics


def train_alphazero(
    model: GrobnerAlphaZero,
    replay_buffer: ReplayBuffer,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[GrobnerAlphaZero, optax.OptState, dict]:
    """
    Train the AlphaZero model on replay buffer data.

    Args:
        model: The GrobnerAlphaZero model.
        replay_buffer: Buffer containing self-play experiences.
        train_config: Training configuration.
        optimizer: Optax optimizer.
        opt_state: Current optimizer state.

    Returns:
        Tuple of (trained_model, new_opt_state, metrics).
    """

    @eqx.filter_jit
    def make_step(
        model: GrobnerAlphaZero,
        opt_state: optax.OptState,
        observations: dict,
        mcts_policies: Array,
        values: Array,
        loss_mask: Array,
    ) -> tuple[GrobnerAlphaZero, optax.OptState, Array, dict]:
        def loss_fn(m):
            loss, metrics = alphazero_loss(m, observations, mcts_policies, values, loss_mask)
            return loss, metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics

    epoch_metrics = {
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
    }

    num_batches = max(1, len(replay_buffer) // train_config.batch_size)

    for epoch in range(train_config.num_epochs_per_iteration):
        for _ in range(num_batches):
            # Sample batch
            batch = replay_buffer.sample(train_config.batch_size)

            # Batch experiences
            observations, mcts_policies, values, loss_mask = batch_experiences(batch)

            # Convert to JAX arrays
            observations = {k: jnp.array(v) for k, v in observations.items()}
            mcts_policies = jnp.array(mcts_policies)
            values = jnp.array(values)
            loss_mask = jnp.array(loss_mask)

            # Training step
            model, opt_state, loss, metrics = make_step(
                model, opt_state, observations, mcts_policies, values, loss_mask
            )

            for k, v in metrics.items():
                epoch_metrics[k].append(float(v))

    # Compute mean metrics
    mean_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

    return model, opt_state, mean_metrics


# =============================================================================
# Main Training Loop
# =============================================================================


def evaluate_model(
    model: GrobnerAlphaZero,
    env: BuchbergerEnv,
    num_episodes: int = 20,
) -> dict:
    """
    Evaluate the model by playing episodes greedily.

    Args:
        model: The GrobnerAlphaZero model.
        env: Environment for evaluation.
        num_episodes: Number of episodes to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    from grobnerRl.envs.env import make_obs

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

            # Greedy action selection
            policy_logits = np.array(policy_logits)

            # Mask invalid actions
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


def alphazero_training_loop(
    model: GrobnerAlphaZero,
    env: BuchbergerEnv,
    num_iterations: int,
    episodes_per_iteration: int,
    mcts_config: MCTSConfig,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    replay_buffer: ReplayBuffer,
    checkpoint_dir: str | None = None,
    eval_interval: int = 5,
    eval_episodes: int = 20,
) -> GrobnerAlphaZero:
    """
    Main AlphaZero training loop.

    Alternates between self-play data generation and training.

    Args:
        model: Initial GrobnerAlphaZero model.
        env: Environment for self-play.
        num_iterations: Number of training iterations.
        episodes_per_iteration: Self-play episodes per iteration.
        mcts_config: MCTS configuration.
        train_config: Training configuration.
        optimizer: Optax optimizer.
        replay_buffer: Replay buffer for storing experiences.
        checkpoint_dir: Directory for saving checkpoints (None to disable).
        eval_interval: Evaluate every N iterations.
        eval_episodes: Number of episodes for evaluation.

    Returns:
        Trained GrobnerAlphaZero model.
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_reward = float("-inf")

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # 1. Self-play: generate data with current model
        print("\nGenerating self-play data...")
        experiences = generate_self_play_data(
            model, env, episodes_per_iteration, mcts_config
        )
        print(f"Generated {len(experiences)} experiences from {episodes_per_iteration} episodes")

        # 2. Add to replay buffer
        replay_buffer.add(experiences)
        print(f"Replay buffer size: {len(replay_buffer)}")

        # 3. Train model on buffer samples
        metrics: dict = {}
        if len(replay_buffer) >= train_config.batch_size:
            print("\nTraining...")
            model, opt_state, metrics = train_alphazero(
                model, replay_buffer, train_config, optimizer, opt_state
            )
            print(
                f"  Policy loss: {metrics['policy_loss']:.4f}, "
                f"Value loss: {metrics['value_loss']:.4f}, "
                f"Total loss: {metrics['total_loss']:.4f}"
            )

            # Save checkpoint
            if checkpoint_dir:
                save_checkpoint(
                    model, opt_state, checkpoint_dir, "last", iteration + 1, metrics
                )

        # 4. Periodic evaluation
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


# =============================================================================
# Main Script
# =============================================================================


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Environment configuration
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)

    # Pretrained model (set to None to train from scratch)
    pretrained_checkpoint_path: str | None = "models/checkpoints/best.eqx"

    # Model configuration
    model_config = ModelConfig(
        monomials_dim=num_vars + 1,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=4,
        ideal_num_heads=8,
        value_hidden_dim=128,
    )

    # MCTS configuration
    mcts_config = MCTSConfig(
        num_simulations=25,  # Reduced for faster iteration
        c_puct=1.0,
        gamma=0.99,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    # Training configuration
    train_config = TrainConfig(
        learning_rate=1e-4,
        batch_size=32,
        num_epochs_per_iteration=3,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    )

    # AlphaZero loop configuration
    num_iterations = 50
    episodes_per_iteration = 5
    replay_buffer_size = 50000
    checkpoint_dir = os.path.join("models", "alphazero_checkpoints")
    eval_interval = 5
    eval_episodes = 10

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Random key
    key = jax.random.key(42)

    # Create environment
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_gen, mode="train")

    # Create optimizer
    optimizer = optax.nadam(train_config.learning_rate)

    # Initialize model
    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained model from {pretrained_checkpoint_path}")
        key, k_model = jax.random.split(key)
        model = GrobnerAlphaZero.from_pretrained(
            checkpoint_path=pretrained_checkpoint_path,
            config=model_config,
            optimizer=optimizer,
            key=k_model,
        )
        print("Pretrained policy loaded. Fresh value head initialized.")
    else:
        print("Initializing model from scratch")
        key, k_model = jax.random.split(key)
        model = GrobnerAlphaZero.from_scratch(
            config=model_config,
            key=k_model,
        )

    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("\nStarting AlphaZero training...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Episodes per iteration: {episodes_per_iteration}")
    print(f"  MCTS simulations: {mcts_config.num_simulations}")
    print(f"  Replay buffer size: {replay_buffer_size}")
    print(f"  Checkpoint directory: {checkpoint_dir}")

    trained_model = alphazero_training_loop(
        model=model,
        env=env,
        num_iterations=num_iterations,
        episodes_per_iteration=episodes_per_iteration,
        mcts_config=mcts_config,
        train_config=train_config,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        checkpoint_dir=checkpoint_dir,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    print("\nDone!")

