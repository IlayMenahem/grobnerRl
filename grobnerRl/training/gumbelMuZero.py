"""Gumbel AlphaZero for the Buchberger environment.

Implements the algorithm from Danihelka et al., "Policy Improvement by Planning
with Gumbel" (ICLR 2022). Because the Buchberger environment is a perfect
deterministic simulator (spoly + reduce + update), this is the AlphaZero
variant: tree nodes hold real (G, P) states, no learned dynamics.

Action spaces are variable: each node has its own action count k = len(P).
All per-node arrays (priors, Gumbels, visit counts, Q sums) are sized to k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sympy.polys.rings import PolyElement


def _np_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for small numpy vectors used in MCTS."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    shifted = x - np.max(x)
    e = np.exp(shifted)
    return (e / e.sum()).astype(np.float32, copy=False)

from grobnerRl.env import BuchbergerEnv, make_obs, reduce, spoly, update
from grobnerRl.ideals import IdealGenerator
from grobnerRl.models import GrobnerPolicyValue
from grobnerRl.training.shared import (
    Experience,
    GrainReplayBuffer,
    TrainConfig,
    train_policy_value,
)


@dataclass
class GumbelAZConfig:
    """Hyperparameters for Gumbel AlphaZero."""

    num_simulations: int = 16
    max_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    discount: float = 1.0
    max_episode_steps: int = 200


def _transition(
    G: list[PolyElement],
    P: list[tuple[int, int]],
    action: tuple[int, int],
) -> tuple[list[PolyElement], list[tuple[int, int]], int, bool]:
    """Pure transition function: run one Buchberger step from (G, P)."""
    G = list(G)
    P = list(P)
    P.remove(action)

    poly = spoly(G[action[0]], G[action[1]])
    poly, stats = reduce(poly, G)
    if poly != 0:
        G, P = update(G, P, poly.monic())

    reward = -(1 + stats["steps"])
    terminated = len(P) == 0
    return G, P, reward, terminated


@dataclass
class Node:
    """A node in the search tree for variable action spaces."""

    G: list[PolyElement]
    P: list[tuple[int, int]]
    is_terminal: bool
    prior_logits: np.ndarray
    value: float
    visit_counts: np.ndarray
    value_sums: np.ndarray
    rewards: np.ndarray
    terminal_child: np.ndarray
    children: dict[int, "Node"] = field(default_factory=dict)


def _evaluate(
    G: list[PolyElement],
    P: list[tuple[int, int]],
    model: GrobnerPolicyValue,
) -> tuple[np.ndarray, float]:
    """Run the model on (G, P) and return per-pair logits and scalar value."""
    obs = make_obs(G, P)
    policy_logits, value = model(obs)

    flat = np.asarray(policy_logits)
    n_padded = int(round(math.sqrt(flat.size)))
    scores = flat.reshape(n_padded, n_padded)
    per_pair = np.array(
        [scores[i, j] for (i, j) in P],
        dtype=np.float32,
    )
    return per_pair, float(value)


def _make_node(
    G: list[PolyElement],
    P: list[tuple[int, int]],
    model: GrobnerPolicyValue,
    is_terminal: bool = False,
) -> Node:
    """Build a tree node by evaluating the model at (G, P)."""
    k = len(P)
    if is_terminal or k == 0:
        return Node(
            G=G,
            P=P,
            is_terminal=True,
            prior_logits=np.zeros(0, dtype=np.float32),
            value=0.0,
            visit_counts=np.zeros(0, dtype=np.float32),
            value_sums=np.zeros(0, dtype=np.float32),
            rewards=np.zeros(0, dtype=np.float32),
            terminal_child=np.zeros(0, dtype=bool),
        )

    logits, value = _evaluate(G, P, model)
    return Node(
        G=G,
        P=P,
        is_terminal=False,
        prior_logits=logits,
        value=value,
        visit_counts=np.zeros(k, dtype=np.float32),
        value_sums=np.zeros(k, dtype=np.float32),
        rewards=np.zeros(k, dtype=np.float32),
        terminal_child=np.zeros(k, dtype=bool),
    )


def _sigma(
    q: np.ndarray, max_visit: float, c_visit: float, c_scale: float
) -> np.ndarray:
    """Q-value scaling transform from equation (8) of the paper."""
    return (c_visit + max_visit) * c_scale * q


def _q_estimates(node: Node) -> np.ndarray:
    """Mean Q per action; 0 where the action has not been visited."""
    safe_counts = np.maximum(node.visit_counts, 1.0)
    q = node.value_sums / safe_counts
    return np.where(node.visit_counts > 0, q, 0.0).astype(np.float32)


def _mixed_value(node: Node) -> float:
    """v_mix approximation of v_pi from Appendix D."""
    if node.prior_logits.size == 0:
        return node.value

    pi = _np_softmax(node.prior_logits)
    visited = node.visit_counts > 0
    sum_n = float(node.visit_counts.sum())

    if sum_n == 0.0 or not visited.any():
        return float(node.value)

    pi_visited_sum = float(pi[visited].sum()) + 1e-9
    q = node.value_sums[visited] / np.maximum(node.visit_counts[visited], 1.0)
    weighted_q = float((pi[visited] * q).sum()) / pi_visited_sum
    return float((node.value + sum_n * weighted_q) / (1.0 + sum_n))


def _completed_q(node: Node, v_mix: float) -> np.ndarray:
    """Completed Q-values (eq. 10) with v_mix filling unvisited actions."""
    q = _q_estimates(node)
    return np.where(node.visit_counts > 0, q, v_mix).astype(np.float32)


def _improved_policy(node: Node, cfg: GumbelAZConfig) -> np.ndarray:
    """Improved policy pi' = softmax(logits + sigma(completedQ)), eq. (11)."""
    if node.prior_logits.size == 0:
        return np.zeros(0, dtype=np.float32)

    v_mix = _mixed_value(node)
    cq = _completed_q(node, v_mix)
    max_visit = float(node.visit_counts.max()) if node.visit_counts.size > 0 else 0.0
    transformed = _sigma(cq, max_visit, cfg.c_visit, cfg.c_scale)
    logits = node.prior_logits + transformed
    return _np_softmax(logits)


def _select_non_root_action(node: Node, cfg: GumbelAZConfig) -> int:
    """Deterministic non-root selection (eq. 14)."""
    pi = _improved_policy(node, cfg)
    sum_n = float(node.visit_counts.sum())
    score = pi - node.visit_counts / (1.0 + sum_n)
    return int(np.argmax(score))


def _simulate(
    root: Node,
    root_action: int,
    model: GrobnerPolicyValue,
    cfg: GumbelAZConfig,
) -> None:
    """One MCTS simulation: descend from root through root_action, expand, backup."""
    path: list[tuple[Node, int]] = []
    node = root
    a = root_action

    while True:
        path.append((node, a))
        if a in node.children:
            child = node.children[a]
            if child.is_terminal:
                bootstrap = 0.0
                break
            node = child
            a = _select_non_root_action(node, cfg)
        else:
            action_pair = node.P[a]
            G_new, P_new, reward, terminated = _transition(node.G, node.P, action_pair)
            child = _make_node(G_new, P_new, model, is_terminal=terminated)
            node.children[a] = child
            node.rewards[a] = reward
            node.terminal_child[a] = terminated
            bootstrap = 0.0 if terminated else child.value
            break

    g_return = bootstrap
    for parent, parent_a in reversed(path):
        r = float(parent.rewards[parent_a])
        g_return = r + cfg.discount * g_return
        parent.visit_counts[parent_a] += 1.0
        parent.value_sums[parent_a] += g_return


def _argtop(values: np.ndarray, n: int) -> list[int]:
    """Indices of the n largest entries of values, sorted descending."""
    n = min(n, values.size)
    order = np.argsort(-values)
    return order[:n].tolist()


def _sequential_halving_with_gumbel(
    root: Node,
    model: GrobnerPolicyValue,
    cfg: GumbelAZConfig,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, float, np.ndarray]:
    """Algorithm 2: pick a root action via Sequential Halving with Gumbel."""
    k = len(root.P)
    assert k > 0, "root must have at least one valid pair"

    gumbels = rng.gumbel(size=k).astype(np.float32)
    m = min(cfg.max_considered_actions, k)
    considered = _argtop(gumbels + root.prior_logits, m)

    n = cfg.num_simulations
    num_phases = max(1, int(math.ceil(math.log2(max(m, 2)))))

    remaining = considered
    budget_used = 0
    while budget_used < n and len(remaining) > 1:
        per_action = max(1, n // (num_phases * len(remaining)))
        for a in remaining:
            for _ in range(per_action):
                if budget_used >= n:
                    break
                _simulate(root, a, model, cfg)
                budget_used += 1
            if budget_used >= n:
                break

        max_visit = float(root.visit_counts.max())
        q = _q_estimates(root)
        rank = gumbels + root.prior_logits + _sigma(
            q, max_visit, cfg.c_visit, cfg.c_scale
        )
        remaining = sorted(remaining, key=lambda a: -rank[a])
        remaining = remaining[: max(1, len(remaining) // 2)]

    while budget_used < n and remaining:
        for a in remaining:
            if budget_used >= n:
                break
            _simulate(root, a, model, cfg)
            budget_used += 1

    max_visit = float(root.visit_counts.max()) if root.visit_counts.size > 0 else 0.0
    q = _q_estimates(root)
    rank = gumbels + root.prior_logits + _sigma(
        q, max_visit, cfg.c_visit, cfg.c_scale
    )
    best_action = max(remaining, key=lambda a: rank[a])

    improved_pi = _improved_policy(root, cfg)
    root_value = _mixed_value(root)

    return best_action, improved_pi, root_value, gumbels


def _build_dense_policy(
    improved_pi: np.ndarray,
    pairs: list[tuple[int, int]],
    num_polys: int,
) -> np.ndarray:
    """Scatter the per-pair improved policy into a dense (num_polys^2,) target."""
    dense = np.zeros(num_polys * num_polys, dtype=np.float32)
    for prob, (i, j) in zip(improved_pi, pairs):
        dense[i * num_polys + j] = float(prob)
    return dense


def run_episode(
    env: BuchbergerEnv,
    model: GrobnerPolicyValue,
    cfg: GumbelAZConfig,
    rng: np.random.Generator,
    seed: Optional[int] = None,
) -> tuple[list[Experience], float, int]:
    """Run one self-play episode with Gumbel AlphaZero search at every step."""
    env.reset(seed=seed)

    samples: list[tuple[Experience, float]] = []
    total_reward = 0.0
    steps = 0

    while steps < cfg.max_episode_steps and len(env.pairs) > 0:
        G = list(env.generators)
        P = list(env.pairs)

        root = _make_node(G, P, model)
        best_idx, improved_pi, root_v, _ = _sequential_halving_with_gumbel(
            root, model, cfg, rng
        )

        num_polys = len(G)
        ideal_tokens = tuple(
            np.asarray(arr, dtype=np.float32) for arr in make_obs(G, P)[0]
        )
        dense_policy = _build_dense_policy(improved_pi, P, num_polys)

        # Placeholder value; we replace with bootstrap returns after the episode.
        exp = Experience(
            ideal=ideal_tokens,
            selectables=tuple(P),
            policy=dense_policy,
            value=root_v,
            num_polys=num_polys,
        )
        samples.append((exp, root_v))

        chosen_pair = root.P[best_idx]
        _, reward, terminated, truncated, _ = env.step(chosen_pair)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return [exp for exp, _ in samples], total_reward, steps


def collect_experiences(
    env: BuchbergerEnv,
    model: GrobnerPolicyValue,
    cfg: GumbelAZConfig,
    num_episodes: int,
    rng: np.random.Generator,
    base_seed: int = 0,
) -> tuple[list[Experience], list[float], list[int]]:
    """Run a batch of self-play episodes and collect experiences."""
    all_experiences: list[Experience] = []
    rewards: list[float] = []
    lengths: list[int] = []

    for ep_idx in range(num_episodes):
        exps, total_reward, steps = run_episode(
            env, model, cfg, rng, seed=base_seed + ep_idx
        )
        all_experiences.extend(exps)
        rewards.append(total_reward)
        lengths.append(steps)

    return all_experiences, rewards, lengths


def train_iteration(
    env: BuchbergerEnv,
    model: GrobnerPolicyValue,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    replay: GrainReplayBuffer,
    cfg: GumbelAZConfig,
    train_cfg: TrainConfig,
    num_episodes: int,
    rng: np.random.Generator,
    base_seed: int,
) -> tuple[GrobnerPolicyValue, optax.OptState, dict]:
    """Run one outer iteration: self-play -> push to replay -> train."""
    experiences, rewards, lengths = collect_experiences(
        env, model, cfg, num_episodes, rng, base_seed=base_seed
    )
    replay.add(experiences)

    if len(replay) == 0:
        return model, opt_state, {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "num_experiences": 0,
        }

    model, opt_state, train_metrics = train_policy_value(
        model, replay, train_cfg, optimizer, opt_state
    )

    metrics = {
        **train_metrics,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "num_experiences": len(experiences),
    }
    return model, opt_state, metrics
