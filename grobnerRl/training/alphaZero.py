"""
AlphaZero training module for the Buchberger environment.

This module implements the AlphaZero algorithm, which uses PUCT-based MCTS
with the real environment for simulation (no learned dynamics model).

Key components:
    - PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    - Dirichlet noise: Added to root priors for exploration during self-play
    - Policy target: Normalized visit-count distribution over valid actions
    - Value target: Discounted return from self-play episode
    - Training loss: Cross-entropy(visit counts, logits) + Huber(value)

References:
    - "Mastering Chess and Shogi by Self-Play with a General Reinforcement
       Learning Algorithm" (Silver et al., 2017)
"""

from dataclasses import dataclass, field

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from grobnerRl.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue
from grobnerRl.training.shared import (
    Experience,
    MinMaxStats,
    copy_env,
    get_valid_actions,
)


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero search."""

    num_simulations: int = 50
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    gamma: float = 0.99


@dataclass
class AlphaZeroNode:
    """
    A node in the AlphaZero MCTS search tree.

    Each node stores the environment state, neural network outputs, and
    MCTS statistics. The Q-value on a child edge is the mean backed-up
    return seen through that edge.

    Attributes:
        env: Copy of the environment at this state (None for unexpanded nodes).
        visit_count: N(s, a) — number of times this edge was traversed.
        value_sum: W(s, a) — sum of backed-up values through this edge.
        reward: Immediate reward received when transitioning to this node.
        prior: P(s, a) — prior probability from the parent's policy network.
        is_terminal: Whether this is a terminal state.
        children: Maps flattened action index to child AlphaZeroNode.
        policy_logits: Raw policy logits from the network at this state.
        network_value: Value estimate from the network at this state.
        valid_actions: Flattened action indices valid at this state.
        num_polys: Number of polynomials in the ideal at this state.
    """

    env: BuchbergerEnv | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    children: dict[int, "AlphaZeroNode"] = field(default_factory=dict)
    policy_logits: np.ndarray | None = None
    network_value: float = 0.0
    valid_actions: list[int] = field(default_factory=list)
    num_polys: int = 0

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        """Whether this node has been expanded (environment state available)."""
        return self.env is not None


def _expand_node(
    node: AlphaZeroNode,
    model: GrobnerPolicyValue,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> None:
    """
    Evaluate the neural network at a node and initialise children with prior probabilities.

    Optionally adds Dirichlet noise to the root priors to encourage exploration
    during self-play, as per the AlphaZero paper.

    Args:
        node: The node to expand. Must have node.env set.
        model: Neural network model for policy and value estimation.
        add_dirichlet_noise: Whether to mix Dirichlet noise into the priors (root only).
        dirichlet_alpha: Concentration parameter for Dirichlet noise.
        dirichlet_epsilon: Weight of Dirichlet noise vs prior probability.
    """
    if node.is_terminal:
        node.network_value = 0.0
        node.valid_actions = []
        return

    if node.env is None:
        raise ValueError("_expand_node requires node.env to be set")

    obs = make_obs(node.env.generators, node.env.pairs)
    policy_logits, value = model(obs)

    node.policy_logits = np.array(policy_logits)
    node.network_value = float(value)
    node.num_polys = len(node.env.generators)
    node.valid_actions = get_valid_actions(node.env)

    if not node.valid_actions:
        return

    # Stable softmax over valid logits to compute prior probabilities
    valid_logits = node.policy_logits[node.valid_actions]
    exp_logits = np.exp(valid_logits - valid_logits.max())
    priors = exp_logits / exp_logits.sum()

    if add_dirichlet_noise:
        noise = np.random.dirichlet([dirichlet_alpha] * len(node.valid_actions))
        priors = (1.0 - dirichlet_epsilon) * priors + dirichlet_epsilon * noise

    for action, prior in zip(node.valid_actions, priors):
        if action not in node.children:
            node.children[action] = AlphaZeroNode(prior=float(prior))
        else:
            node.children[action].prior = float(prior)


def _puct_score(
    node: AlphaZeroNode, child: AlphaZeroNode, c_puct: float, min_max_stats: MinMaxStats
) -> float:
    """
    Compute the PUCT score for a child edge.

    Score = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

    Q-values are normalised to [0, 1] using tree-wide min/max statistics so
    the exploration bonus operates on a consistent scale.

    Args:
        node: Parent node (provides total visit count N(s)).
        child: Child node (provides N(s,a), W(s,a), P(s,a)).
        c_puct: Exploration constant.
        min_max_stats: Tree-wide min/max for Q-value normalisation.

    Returns:
        PUCT score for this child edge.
    """
    parent_visits = sum(c.visit_count for c in node.children.values())
    exploration = (
        c_puct * child.prior * np.sqrt(parent_visits) / (1 + child.visit_count)
    )
    normalised_q = min_max_stats.normalize(child.q_value)
    return normalised_q + exploration


def _select_child(
    node: AlphaZeroNode, c_puct: float, min_max_stats: MinMaxStats
) -> int:
    """
    Select the child action with the highest PUCT score.

    Args:
        node: The node to select from. Must be expanded with valid_actions.
        c_puct: Exploration constant for PUCT scoring.
        min_max_stats: Tree-wide min/max statistics for Q normalisation.

    Returns:
        Flattened action index of the selected child.
    """
    best_action = max(
        node.valid_actions,
        key=lambda a: _puct_score(node, node.children[a], c_puct, min_max_stats),
    )
    return best_action


def _backup(
    path: list[tuple[AlphaZeroNode, int]],
    leaf_value: float,
    gamma: float,
    min_max_stats: MinMaxStats,
) -> None:
    """
    Propagate the leaf value up the search path, updating visit counts and value sums.

    For each edge (node, action) -> child traversed from leaf to root:
        G = child.reward + gamma * G
        child.visit_count += 1
        child.value_sum  += G

    Args:
        path: List of (node, action) pairs from root down to the leaf's parent.
        leaf_value: Network value estimate at the leaf (0 if terminal).
        gamma: Discount factor.
        min_max_stats: Tree-wide statistics updated with each new Q-value.
    """
    value = leaf_value
    for node, action in reversed(path):
        child = node.children[action]
        value = child.reward + gamma * value
        child.visit_count += 1
        child.value_sum += value
        min_max_stats.update(child.q_value)


def _run_simulation(
    root: AlphaZeroNode,
    model: GrobnerPolicyValue,
    config: AlphaZeroConfig,
    min_max_stats: MinMaxStats,
) -> None:
    """
    Run one MCTS simulation from the root to a leaf, then backup.

    Traverses the tree by selecting children via PUCT until reaching an
    unexpanded or terminal node. The leaf is expanded with the neural
    network and values are backed up to the root.

    Args:
        root: Root node of the search tree (must already be expanded).
        model: Neural network for evaluating newly expanded leaf nodes.
        config: AlphaZero configuration (c_puct, gamma).
        min_max_stats: Tree-wide min/max statistics shared across simulations.
    """
    path: list[tuple[AlphaZeroNode, int]] = []
    node = root

    while node.expanded and not node.is_terminal:
        action = _select_child(node, config.c_puct, min_max_stats)
        path.append((node, action))

        if action not in node.children:
            node.children[action] = AlphaZeroNode(prior=0.0)
        child = node.children[action]

        if not child.expanded:
            # Step the real environment and expand the leaf
            child.env = copy_env(node.env)
            i, j = action // node.num_polys, action % node.num_polys
            _, reward, terminated, _, _ = child.env.step((i, j))
            child.reward = float(reward)
            child.is_terminal = bool(terminated)
            child.num_polys = len(child.env.generators)

            if terminated:
                child.network_value = 0.0
                child.valid_actions = []
                _backup(path, 0.0, config.gamma, min_max_stats)
            else:
                _expand_node(child, model)
                _backup(path, child.network_value, config.gamma, min_max_stats)
            return

        if child.is_terminal:
            _backup(path, 0.0, config.gamma, min_max_stats)
            return

        node = child

    # Reached a terminal root (edge case)
    _backup(path, 0.0, config.gamma, min_max_stats)


def _build_policy_target(root: AlphaZeroNode) -> np.ndarray:
    """
    Construct the policy training target from root visit counts.

    The target is the normalised visit-count distribution over all positions
    in the flattened action space (zeros for invalid actions).

    Args:
        root: Expanded root node after all simulations have been run.

    Returns:
        Dense policy array of shape (num_polys * num_polys,) summing to 1
        over valid actions.
    """
    action_space_size = root.num_polys * root.num_polys
    visit_counts = np.zeros(action_space_size, dtype=np.float32)

    for action in root.valid_actions:
        if action in root.children:
            visit_counts[action] = float(root.children[action].visit_count)

    total = visit_counts.sum()
    if total > 0:
        visit_counts /= total

    return visit_counts


class AlphaZeroSearch:
    """
    AlphaZero MCTS search algorithm.

    Runs PUCT-guided tree search using the real environment for transitions.
    After all simulations, the policy target is the normalised visit-count
    distribution and the value estimate is the root network value.
    """

    def __init__(self, model: GrobnerPolicyValue, config: AlphaZeroConfig):
        """
        Args:
            model: Neural network model for policy and value estimation.
            config: AlphaZero search configuration.
        """
        self.model = model
        self.config = config

    def search(self, env: BuchbergerEnv, key: PRNGKeyArray) -> tuple[np.ndarray, float]:
        """
        Run AlphaZero MCTS from the current environment state.

        1. Expand root with Dirichlet-noised priors.
        2. Run num_simulations PUCT simulations.
        3. Return normalised visit counts as the policy target and the
           root network value as the value estimate.

        Args:
            env: Current environment state (not mutated).
            key: JAX random key (unused — numpy RNG used for Dirichlet).

        Returns:
            Tuple of:
                - policy_target: Normalised visit-count vector of shape
                  (num_polys * num_polys,).
                - value: Root network value estimate.
        """
        num_polys = len(env.generators)

        root = AlphaZeroNode(env=copy_env(env))
        root.num_polys = num_polys
        _expand_node(
            root,
            self.model,
            add_dirichlet_noise=True,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

        if not root.valid_actions:
            return np.zeros(num_polys * num_polys, dtype=np.float32), root.network_value

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            _run_simulation(root, self.model, self.config, min_max_stats)

        policy_target = _build_policy_target(root)
        return policy_target, root.network_value


def run_self_play_episode(
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    config: AlphaZeroConfig,
    key: PRNGKeyArray,
) -> list[Experience]:
    """
    Run a single self-play episode using AlphaZero MCTS.

    At each step, runs MCTS to obtain a policy target (visit counts) and
    acts by sampling from that distribution. Experiences are collected with
    placeholder value 0.0 and then updated with discounted returns once the
    episode ends.

    Args:
        model: Neural network model.
        env: Environment for the episode (reset internally).
        config: AlphaZero configuration.
        key: JAX random key for reproducible action sampling.

    Returns:
        List of Experience objects with policy targets and discounted returns.
    """
    env.reset()
    search = AlphaZeroSearch(model, config)

    experiences: list[Experience] = []
    rewards: list[float] = []
    done = False

    while not done:
        current_obs = make_obs(env.generators, env.pairs)
        num_polys = len(env.generators)

        key, subkey = jax.random.split(key)
        policy_target, _ = search.search(env, subkey)

        ideal, selectables = current_obs
        exp = Experience(
            ideal=tuple(
                poly.astype(np.float32) if poly.dtype != np.float32 else poly
                for poly in ideal
            ),
            selectables=tuple(tuple(pair) for pair in selectables)
            if isinstance(selectables, list)
            else selectables,
            policy=policy_target.astype(np.float32),
            value=0.0,
            num_polys=num_polys,
        )
        experiences.append(exp)

        # Sample action from the visit-count policy
        total = policy_target.sum()
        if total > 0:
            action_probs = policy_target / total
        else:
            valid_actions = get_valid_actions(env)
            action_probs = np.zeros_like(policy_target)
            for a in valid_actions:
                action_probs[a] = 1.0 / len(valid_actions)

        action = int(np.random.choice(len(action_probs), p=action_probs))
        i, j = action // num_polys, action % num_polys

        _, reward, terminated, truncated, _ = env.step((i, j))
        rewards.append(float(reward))
        done = terminated or truncated

    # Assign discounted returns as value targets
    G = 0.0
    returns: list[float] = []
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
    config: AlphaZeroConfig,
    key: PRNGKeyArray,
) -> list[Experience]:
    """
    Generate self-play experience from multiple episodes.

    Args:
        model: Neural network model.
        env: Environment used for all episodes.
        num_episodes: Number of self-play episodes to run.
        config: AlphaZero configuration.
        key: JAX random key, split across episodes.

    Returns:
        Flat list of all Experience objects from all episodes.
    """
    from tqdm import tqdm

    all_experiences: list[Experience] = []

    for _ in tqdm(range(num_episodes), desc="Self-play"):
        key, subkey = jax.random.split(key)
        experiences = run_self_play_episode(model, env, config, subkey)
        all_experiences.extend(experiences)

    return all_experiences
