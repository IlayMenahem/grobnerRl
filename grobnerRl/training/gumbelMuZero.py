"""
Gumbel MuZero implementation for the Buchberger environment.

This module implements the Gumbel MuZero algorithm which uses Gumbel-Top-k
sampling with Sequential Halving and full MCTS tree search for efficient
action selection with provable policy improvement.

Key components aligned with the paper:
    - Gumbel-Top-k sampling: Sample m actions without replacement (Equations 3-5)
    - Sequential Halving: Iteratively halve action set, scoring by g(a) + logits(a) + σ(q̂(a))
    - Deep MCTS: Full tree search with non-root action selection (Section 5, Equation 14)
    - Sigma transformation: σ(q̂(a)) = (c_visit + max_b N(b)) * c_scale * q̂(a) (Equation 8)
    - Q-value normalization: Min-max normalization across the search tree
    - v_mix estimator: Interpolates network value with prior-weighted Q-values (Equation 33)
    - Completed Q-values: completedQ(a) = q(a) if visited, else v_mix (Equation 10)
    - Improved policy: π' = softmax(logits + σ(completedQ)) (Equation 11)
    - Policy training: Cross-entropy with improved policy target (Equation 12)

References:
    - "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)
"""

from copy import copy
from dataclasses import dataclass, field
from typing import Literal, overload

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


class MinMaxStats:
    """
    Tracks min/max Q-values across the entire search tree for normalization.

    Q-values are normalized to [0, 1] using the global min/max found during
    tree search, as described in the MuZero paper. This ensures the sigma
    transformation operates on a consistent scale regardless of the
    environment's reward magnitude.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class GumbelNode:
    """
    A node in the Gumbel MuZero search tree.

    Each node stores the environment state, neural network outputs, and
    MCTS statistics. The Q-value stored on a child node represents
    Q(parent_state, action_to_child).

    Attributes:
        env: Copy of the environment at this state (None for unexpanded nodes).
        visit_count: Number of times this node has been visited during backup.
        value_sum: Sum of backed-up values (Q-value of parent->this edge).
        reward: Immediate reward received when transitioning to this node.
        prior: Prior probability from parent's policy network.
        is_terminal: Whether this is a terminal state.
        children: Dictionary mapping action indices to child nodes.
        policy_logits: Raw policy logits from neural network at this state.
        network_value: Value estimate from the neural network at this state.
        valid_actions: List of valid flattened action indices at this state.
        num_polys: Number of polynomials in the ideal at this state.
    """

    env: BuchbergerEnv | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    children: dict[int, "GumbelNode"] = field(default_factory=dict)
    policy_logits: np.ndarray | None = None
    network_value: float = 0.0
    valid_actions: list[int] = field(default_factory=list)
    num_polys: int = 0

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        """Whether this node has been expanded (environment state available)."""
        return self.env is not None


def sigma(
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """
    Compute the sigma transformation of Q-values as per Equation 8 in the paper.

    σ(q̂(a)) = (c_visit + max_b N(b)) * c_scale * q̂(a)

    The progressive scaling increases the influence of Q-values as more
    visits accumulate, reducing the effect of the prior policy.

    Args:
        q_values: Q-values for each action (should be normalized to [0, 1]).
        visit_counts: Visit counts for each action.
        c_visit: Visit count scaling constant (default: 50).
        c_scale: Q-value scaling constant (default: 1.0).

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
    Sample top-k actions without replacement using Gumbel-Top-k trick (Eqs. 3-5).

    Samples k Gumbel variables, adds them to logits, and takes the top-k.
    The same Gumbel values are reused later for action selection to ensure
    the policy improvement guarantee.

    Args:
        key: JAX random key.
        logits: Policy logits for all actions.
        valid_actions: List of valid action indices.
        k: Number of actions to sample.

    Returns:
        Tuple of (selected_actions, gumbel_values) for the top-k actions.
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


def get_valid_actions(env: BuchbergerEnv) -> list[int]:
    """Get list of valid actions as flattened pair indices."""
    num_polys = len(env.generators)
    return [i * num_polys + j for i, j in env.pairs]


def expand_node(node: GumbelNode, model: GrobnerPolicyValue) -> None:
    """
    Evaluate the neural network at a node and initialize children with priors.

    Assumes node.env is already set. Computes policy logits and value estimate,
    determines valid actions, and creates skeleton child nodes with prior
    probabilities from the policy network.

    Args:
        node: The node to expand. Must have node.env set.
        model: Neural network model for policy and value estimation.
    """
    if node.is_terminal:
        node.network_value = 0.0
        node.valid_actions = []
        return

    if node.env is None:
        raise ValueError("expand_node requires node.env to be set")

    obs = make_obs(node.env.generators, node.env.pairs)
    policy_logits, value = model(obs)

    node.policy_logits = np.array(policy_logits)
    node.network_value = float(value)
    node.num_polys = len(node.env.generators)
    node.valid_actions = get_valid_actions(node.env)

    # Initialize children with priors from softmax over valid logits
    if node.valid_actions:
        valid_logits = node.policy_logits[node.valid_actions]
        max_logit = valid_logits.max()
        exp_logits = np.exp(valid_logits - max_logit)
        probs = exp_logits / exp_logits.sum()
        for va, p in zip(node.valid_actions, probs):
            if va not in node.children:
                node.children[va] = GumbelNode(prior=float(p))
            else:
                node.children[va].prior = float(p)


def compute_v_mix(node: GumbelNode) -> float:
    """
    Compute the v_mix value estimate as per Equation 33 (Appendix D).

    v_mix = (v̂_π + (Σ_b N(b) / Σ_{b:N(b)>0} π(b)) · Σ_{a:N(a)>0} π(a)q(a)) / (1 + Σ_b N(b))

    This estimator interpolates between the network's value estimate v̂_π
    (when few visits) and the prior-weighted average of visited Q-values
    (when many visits). It provides a consistent estimate of v_π for
    completing Q-values of unvisited actions.

    Args:
        node: An expanded node with valid_actions and children.

    Returns:
        The v_mix value estimate.
    """
    total_visits = sum(
        node.children[a].visit_count
        for a in node.valid_actions
        if a in node.children
    )

    if total_visits == 0:
        return node.network_value

    visited_prior_sum = sum(
        node.children[a].prior
        for a in node.valid_actions
        if a in node.children and node.children[a].visit_count > 0
    )
    weighted_q_sum = sum(
        node.children[a].prior * node.children[a].q_value
        for a in node.valid_actions
        if a in node.children and node.children[a].visit_count > 0
    )

    if visited_prior_sum > 0:
        v_mix = (
            node.network_value
            + (total_visits / visited_prior_sum) * weighted_q_sum
        ) / (1 + total_visits)
    else:
        v_mix = node.network_value

    return v_mix


def select_non_root_action(
    node: GumbelNode,
    config: GumbelMuZeroConfig,
    min_max_stats: MinMaxStats,
) -> int | None:
    """
    Select action at a non-root node using Equation 14 from the paper (Section 5).

    Computes the improved policy π' = softmax(logits + σ(completedQ)) using
    completed Q-values, then selects the action that minimizes the MSE between
    π' and the normalized visit counts:

        argmax_a (π'(a) - N(a) / (1 + Σ_b N(b)))

    This deterministic selection ensures visit counts converge to the improved
    policy, avoiding the variance of stochastic sampling at non-root nodes.

    Args:
        node: The non-root node to select an action for.
        config: Gumbel MuZero configuration.
        min_max_stats: Tree-wide min/max statistics for Q-value normalization.

    Returns:
        Selected action index, or None if no valid actions.
    """
    if not node.valid_actions or node.policy_logits is None:
        return None

    v_mix = compute_v_mix(node)

    # Build completed Q-values (Eq. 10 with v_mix)
    completed_q = {}
    for a in node.valid_actions:
        if a in node.children and node.children[a].visit_count > 0:
            completed_q[a] = node.children[a].q_value
        else:
            completed_q[a] = v_mix

    # Normalize Q-values using tree-wide min-max stats
    normalized_q = {a: min_max_stats.normalize(q) for a, q in completed_q.items()}

    # Compute sigma transformation
    visit_counts = np.array([
        node.children[a].visit_count if a in node.children else 0
        for a in node.valid_actions
    ])
    max_visit = visit_counts.max() if len(visit_counts) > 0 else 0
    scale = (config.c_visit + max_visit) * config.c_scale

    # Compute improved policy π' = softmax(logits + σ(normalized_completedQ))
    improved_logits = np.array([
        node.policy_logits[a] + scale * normalized_q[a]
        for a in node.valid_actions
    ])
    max_l = improved_logits.max()
    exp_l = np.exp(improved_logits - max_l)
    pi_prime = exp_l / exp_l.sum()

    # Select argmax_a (π'(a) - N(a) / (1 + Σ_b N(b))) [Eq. 14]
    total_visits = int(visit_counts.sum())

    best_action = None
    best_score = -float("inf")
    for idx, a in enumerate(node.valid_actions):
        vc = visit_counts[idx]
        score = pi_prime[idx] - vc / (1 + total_visits)
        if score > best_score:
            best_score = score
            best_action = a

    return best_action


def backup(
    path: list[tuple[GumbelNode, int]],
    leaf_value: float,
    config: GumbelMuZeroConfig,
    min_max_stats: MinMaxStats,
) -> None:
    """
    Backup values from leaf to root through the search path.

    Propagates the leaf value upward through the tree, computing discounted
    returns at each edge. Updates visit counts, value sums, and the
    tree-wide min/max statistics used for Q-value normalization.

    For each edge (parent, action) -> child on the path (leaf to root):
        G = child.reward + gamma * G
        child.visit_count += 1
        child.value_sum += G

    Args:
        path: List of (node, action) pairs from root to leaf.
        leaf_value: Value estimate at the leaf node (0.0 if terminal).
        config: Gumbel MuZero configuration (for gamma).
        min_max_stats: Tree-wide statistics to update.
    """
    value = leaf_value
    for node, action in reversed(path):
        child = node.children[action]
        value = child.reward + config.gamma * value
        child.visit_count += 1
        child.value_sum += value
        min_max_stats.update(child.q_value)


def run_simulation(
    root: GumbelNode,
    root_action: int,
    model: GrobnerPolicyValue,
    config: GumbelMuZeroConfig,
    min_max_stats: MinMaxStats,
) -> None:
    """
    Run a single MCTS simulation through the search tree.

    Starting from the root, forces root_action at the first level (as
    directed by Sequential Halving), then uses the non-root deterministic
    action selection (Eq. 14) at deeper levels. When a leaf (unexpanded node)
    is reached, it is expanded using the neural network and values are
    backed up through the path.

    Args:
        root: Root node of the search tree.
        root_action: Action forced at the root (from Sequential Halving).
        model: Neural network model for expanding leaf nodes.
        config: Gumbel MuZero configuration.
        min_max_stats: Tree-wide min/max statistics for Q-value normalization.
    """
    path: list[tuple[GumbelNode, int]] = []
    node = root
    action = root_action

    while True:
        path.append((node, action))

        # Get or create child node
        if action not in node.children:
            node.children[action] = GumbelNode(prior=0.0)
        child = node.children[action]

        # Leaf node: expand, then backup
        if not child.expanded:
            child.env = copy_env(node.env)
            i, j = action // node.num_polys, action % node.num_polys
            _, reward, terminated, _, _ = child.env.step((i, j))
            child.reward = reward
            child.is_terminal = terminated
            child.num_polys = len(child.env.generators)

            if terminated:
                child.network_value = 0.0
                child.valid_actions = []
                leaf_value = 0.0
            else:
                expand_node(child, model)
                leaf_value = child.network_value

            backup(path, leaf_value, config, min_max_stats)
            return

        # Terminal node: backup with value 0
        if child.is_terminal:
            backup(path, 0.0, config, min_max_stats)
            return

        # Internal node: select next action using non-root policy (Eq. 14)
        node = child
        next_action = select_non_root_action(node, config, min_max_stats)

        if next_action is None:
            # Safety fallback: no valid actions (should not happen for non-terminal)
            backup(path, node.network_value, config, min_max_stats)
            return

        action = next_action


def sequential_halving(
    actions: np.ndarray,
    gumbel_values: np.ndarray,
    policy_logits: np.ndarray,
    root: GumbelNode,
    model: GrobnerPolicyValue,
    config: GumbelMuZeroConfig,
    min_max_stats: MinMaxStats,
) -> int:
    """
    Sequential Halving with Gumbel (Algorithm 2 from the paper).

    Divides the simulation budget into ceil(log2(m)) phases. In each phase,
    the budget is split equally among remaining actions, running full MCTS
    simulations for each. After each phase, actions are scored by
    g(a) + logits(a) + σ(q̂(a)) and the bottom half is eliminated.

    This procedure efficiently identifies the best root action while ensuring
    the policy improvement guarantee from the Gumbel-Top-k trick.

    Args:
        actions: Initial set of candidate actions (from Gumbel-Top-k).
        gumbel_values: Gumbel values for each candidate action.
        policy_logits: Full policy logits from neural network.
        root: Root node of the search tree (already expanded).
        model: Neural network model for simulations.
        config: Gumbel MuZero configuration.
        min_max_stats: Tree-wide min/max statistics for Q-value normalization.

    Returns:
        Selected action index (A_{n+1}).
    """
    if len(actions) == 1:
        # Still run simulations to build the tree for Q-value estimates
        for _ in range(config.num_simulations):
            run_simulation(root, int(actions[0]), model, config, min_max_stats)
        return int(actions[0])

    remaining_actions = actions.copy()
    remaining_gumbels = gumbel_values.copy()

    num_phases = max(1, int(np.ceil(np.log2(len(actions)))))
    total_budget = config.num_simulations
    sims_used = 0

    for phase in range(num_phases):
        if len(remaining_actions) <= 1:
            break

        # Budget per phase: n / ceil(log2(m))
        phase_budget = total_budget // num_phases
        # At least 1 visit per action per phase
        sims_per_action = max(1, phase_budget // len(remaining_actions))

        # Run MCTS simulations for each remaining action
        for action in remaining_actions:
            for _ in range(sims_per_action):
                if sims_used >= total_budget:
                    break
                run_simulation(root, int(action), model, config, min_max_stats)
                sims_used += 1
            if sims_used >= total_budget:
                break

        # Score remaining actions: g(a) + logits(a) + σ(q̂(a))
        q_values = np.array([
            root.children[int(a)].q_value if int(a) in root.children else 0.0
            for a in remaining_actions
        ])
        visit_counts = np.array([
            root.children[int(a)].visit_count if int(a) in root.children else 0
            for a in remaining_actions
        ])

        # Normalize Q-values using tree-wide min-max stats
        normalized_q = np.array([min_max_stats.normalize(q) for q in q_values])

        action_logits = policy_logits[remaining_actions]
        sigma_values = sigma(
            normalized_q, visit_counts, config.c_visit, config.c_scale
        )
        scores = remaining_gumbels + action_logits + sigma_values

        # Keep top half
        num_to_keep = max(1, len(remaining_actions) // 2)
        top_indices = np.argsort(scores)[-num_to_keep:]
        remaining_actions = remaining_actions[top_indices]
        remaining_gumbels = remaining_gumbels[top_indices]

        if sims_used >= total_budget:
            break

    # Final selection from remaining actions
    if len(remaining_actions) == 1:
        return int(remaining_actions[0])

    # Select action with highest g(a) + logits(a) + σ(q̂(a))
    q_values = np.array([
        root.children[int(a)].q_value if int(a) in root.children else 0.0
        for a in remaining_actions
    ])
    visit_counts = np.array([
        root.children[int(a)].visit_count if int(a) in root.children else 0
        for a in remaining_actions
    ])
    normalized_q = np.array([min_max_stats.normalize(q) for q in q_values])
    action_logits = policy_logits[remaining_actions]
    sigma_values = sigma(
        normalized_q, visit_counts, config.c_visit, config.c_scale
    )
    scores = remaining_gumbels + action_logits + sigma_values
    best_idx = np.argmax(scores)
    return int(remaining_actions[best_idx])


class GumbelMuZeroSearch:
    """
    Gumbel MuZero search algorithm.

    Implements the full Gumbel MuZero planning algorithm:
    1. Expand root node with neural network evaluation
    2. Sample m actions without replacement using Gumbel-Top-k (Eqs. 3-5)
    3. Run Sequential Halving with full MCTS simulations (Algorithm 2)
    4. Construct improved policy π' using completed Q-values (Eqs. 10-11)
    5. Return improved policy as training target and root value estimate
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
            env: Environment (used as template; actual search copies state).
            config: Search configuration.
        """
        self.model = model
        self.env = env
        self.config = config

    @overload
    def search(
        self,
        env: BuchbergerEnv,
        key: PRNGKeyArray,
        return_selected_action: Literal[False] = False,
    ) -> tuple[np.ndarray, float]:
        ...

    @overload
    def search(
        self,
        env: BuchbergerEnv,
        key: PRNGKeyArray,
        return_selected_action: Literal[True],
    ) -> tuple[np.ndarray, float, int | None]:
        ...

    def search(
        self,
        env: BuchbergerEnv,
        key: PRNGKeyArray,
        return_selected_action: bool = False,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, int | None]:
        """
        Run Gumbel MuZero search from current state.

        Performs Gumbel-Top-k sampling, Sequential Halving with full MCTS,
        and constructs the improved policy for training.

        Args:
            env: Current environment state.
            key: JAX random key.
            return_selected_action: If True, also return root action selected by
                Sequential Halving (A_{n+1}) for acting in the environment.

        Returns:
            Tuple of (improved_policy, value):
                - improved_policy: π' = softmax(logits + σ(completedQ)) (Eq. 11)
                - value: Root value estimate from the neural network
            If return_selected_action=True, returns:
                - improved_policy, value, selected_action
        """
        num_polys = len(env.generators)

        # Initialize and expand root node
        root = GumbelNode(env=copy_env(env))
        expand_node(root, self.model)

        value = root.network_value
        policy_logits = root.policy_logits
        valid_actions = root.valid_actions
        if policy_logits is None:
            raise ValueError("Root expansion did not produce policy logits")

        if len(valid_actions) == 0:
            empty_policy = np.zeros(num_polys * num_polys)
            if return_selected_action:
                return empty_policy, value, None
            return empty_policy, value

        # Initialize tree-wide min-max stats for Q-value normalization
        min_max_stats = MinMaxStats()

        # Gumbel-Top-k sampling (Eqs. 3-5)
        k = min(self.config.max_num_considered_actions, len(valid_actions))
        actions, gumbel_values = gumbel_top_k(key, policy_logits, valid_actions, k)

        # Sequential Halving with full MCTS (Algorithm 2)
        selected_action = sequential_halving(
            actions,
            gumbel_values,
            policy_logits,
            root,
            self.model,
            self.config,
            min_max_stats,
        )

        # Construct improved policy using completed Q-values (Eqs. 10-11)
        # Use v_mix (Eq. 33) for unvisited actions
        v_mix = compute_v_mix(root)
        min_max_stats.update(v_mix)

        # Build completed Q-values for valid actions
        completed_q = np.full(num_polys * num_polys, 0.0, dtype=np.float32)
        for action in valid_actions:
            if action in root.children and root.children[action].visit_count > 0:
                completed_q[action] = root.children[action].q_value
            else:
                completed_q[action] = v_mix

        # Normalize Q-values using tree-wide min-max stats
        normalized_completed_q = np.zeros(num_polys * num_polys, dtype=np.float32)
        for action in valid_actions:
            normalized_completed_q[action] = min_max_stats.normalize(
                completed_q[action]
            )

        # Compute sigma transformation
        visit_counts = np.zeros(num_polys * num_polys)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        sigma_values = sigma(
            normalized_completed_q, visit_counts, self.config.c_visit, self.config.c_scale
        )

        # Improved policy: π' = softmax(logits + σ(completedQ))
        improved_logits = policy_logits + sigma_values

        # Mask invalid actions
        mask = np.full_like(improved_logits, -np.inf)
        for action in valid_actions:
            mask[action] = 0.0
        improved_logits = improved_logits + mask

        # Stable softmax over valid actions
        valid_max = improved_logits[valid_actions].max()
        exp_logits = np.exp(improved_logits - valid_max)
        exp_logits = np.where(np.isinf(improved_logits), 0.0, exp_logits)
        policy = exp_logits / exp_logits.sum()

        if return_selected_action:
            return policy, value, selected_action
        return policy, value


def run_self_play_episode(
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    config: GumbelMuZeroConfig,
    key: PRNGKeyArray,
) -> list[Experience]:
    """
    Run a single self-play episode using Gumbel MuZero search.

    At each step, runs the full search to get the improved policy (used as
    training target) and acts with the root action selected by Sequential
    Halving (A_{n+1}).

    Args:
        model: Neural network model.
        env: Environment for the episode.
        config: Gumbel MuZero configuration.
        key: JAX random key.

    Returns:
        List of experiences with improved policies and discounted returns.
    """
    env.reset()
    search = GumbelMuZeroSearch(model, env, config)

    experiences = []
    rewards = []
    done = False

    while not done:
        current_obs = make_obs(env.generators, env.pairs)
        num_polys = len(env.generators)

        key, subkey = jax.random.split(key)
        policy, _, selected_action = search.search(
            env, subkey, return_selected_action=True
        )

        # Create uncompressed experience directly
        ideal, selectables = current_obs
        exp = Experience(
            ideal=tuple(poly.astype(np.float32) if poly.dtype != np.float32 else poly for poly in ideal),
            selectables=tuple(tuple(pair) for pair in selectables) if isinstance(selectables, list) else selectables,
            policy=policy.astype(np.float32),
            value=0.0,
            num_polys=num_polys,
        )
        experiences.append(exp)

        # Act with the selected root action A_{n+1}. Fallback to policy sampling
        # only if no action is selected (e.g., no valid actions).
        if selected_action is not None:
            action = int(selected_action)
        else:
            policy_sum = policy.sum()
            if policy_sum > 0:
                normalized_policy = policy / policy_sum
            else:
                valid_actions = get_valid_actions(env)
                normalized_policy = np.zeros_like(policy)
                for a in valid_actions:
                    normalized_policy[a] = 1.0 / len(valid_actions)
            action = int(np.random.choice(len(policy), p=normalized_policy))
        i, j = action // num_polys, action % num_polys

        _, reward, terminated, truncated, _ = env.step((i, j))
        rewards.append(reward)
        done = terminated or truncated

    # Compute discounted returns for value targets
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
) -> list[Experience]:
    """Generate self-play data from multiple episodes."""
    from tqdm import tqdm

    all_experiences = []

    for episode in tqdm(range(num_episodes), desc="Self-play"):
        key, subkey = jax.random.split(key)
        experiences = run_self_play_episode(model, env, config, subkey)
        all_experiences.extend(experiences)

    return all_experiences
