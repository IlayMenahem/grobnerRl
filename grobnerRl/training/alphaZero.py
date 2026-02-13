"""
AlphaZero-style MCTS training for Buchberger environment.

This script implements:
- MCTS with neural network guidance (policy + value)
- Self-play data generation
- Combined policy and value training
- Support for pretrained models from supervised_jax.py
"""

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
from grobnerRl.models import (
    Extractor,
    GrobnerPolicy,
    IdealModel,
    MonomialEmbedder,
    PairwiseScorer,
    PolynomialEmbedder,
    GrobnerPolicyValue,
)
from grobnerRl.training.shared import (
    Experience,
    ReplayBuffer,
    TrainConfig,
    evaluate_model,
    train_policy_value,
)
from grobnerRl.training.utils import load_checkpoint, save_checkpoint


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================


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



@eqx.filter_jit
def _jit_single_inference(model: GrobnerPolicyValue, obs: tuple) -> tuple[Array, Array]:
    """JIT-compiled inference for a single observation."""
    return model(obs)


@eqx.filter_jit
def _jit_batched_inference(model: GrobnerPolicyValue, batched_obs: dict) -> tuple[Array, Array]:
    """JIT-compiled batched inference using filter_vmap."""
    return eqx.filter_vmap(model)(batched_obs)


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT (Predictor + Upper Confidence bounds for Trees) for selection,
    and a neural network for policy priors and value estimation.
    """

    def __init__(
        self,
        model: GrobnerPolicyValue,
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


def run_self_play_episode(
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    mcts_config: MCTSConfig,
    poly_cache,
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

        exp = Experience.from_uncompressed(
            observation=current_obs,
            policy=policy,
            value=0.0,
            num_polys=num_polys,
            poly_cache=poly_cache,
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
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    num_episodes: int,
    mcts_config: MCTSConfig,
    poly_cache,
) -> list[Experience]:
    """Generate self-play data from multiple episodes."""
    all_experiences = []

    for episode in tqdm(range(num_episodes), desc="Self-play"):
        experiences = run_self_play_episode(model, env, mcts_config, poly_cache, seed=episode)
        all_experiences.extend(experiences)

    return all_experiences


def alphazero_training_loop(
    model: GrobnerPolicyValue,
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
) -> GrobnerPolicyValue:
    """
    Main AlphaZero training loop.

    Alternates between self-play data generation and training.

    Args:
        model: Initial GrobnerPolicyValue model.
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
        Trained GrobnerPolicyValue model.
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
            model, env, episodes_per_iteration, mcts_config, replay_buffer.poly_cache
        )
        print(f"Generated {len(experiences)} experiences from {episodes_per_iteration} episodes")

        # 2. Add to replay buffer
        replay_buffer.add(experiences)
        print(f"Replay buffer size: {len(replay_buffer)}")

        # 3. Train model on buffer samples
        metrics: dict = {}
        if len(replay_buffer) >= train_config.batch_size:
            print("\nTraining...")
            model, opt_state, metrics = train_policy_value(
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
