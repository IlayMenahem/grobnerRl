"""
Rainbow DQN training for the Gröbner basis environment.

Rainbow components implemented:
  - Double DQN         : online net selects action, target net evaluates Q-value.
  - Dueling networks   : existing GrobnerPolicyValue policy-logits are the advantage
                         stream and the value head is the state value; combined they
                         form per-action Q-values.
  - Prioritized Experience Replay (PER) : sum-tree for O(log N) weighted sampling
                         with importance-sampling correction.
  - Multi-step returns : an n-step transition buffer accumulates rewards before
                         pushing experiences into the PER buffer.
  - Target network     : soft (Polyak) updates every step; hard sync every
                         `target_update_interval` steps.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue
from grobnerRl.training.utils import (
    create_metrics_log_path,
    log_metrics,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RainbowConfig:
    """Hyper-parameters for the Rainbow DQN training loop."""

    # Optimisation
    learning_rate: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64

    # Target network
    target_update_interval: int = 200  # hard sync every N gradient steps
    tau: float = 1.0  # Polyak factor (1.0 = hard copy)

    # Replay buffer
    replay_buffer_size: int = 2**14
    min_replay_size: int = 512  # warm-up steps before training starts

    # Multi-step returns
    n_step: int = 3

    # Prioritized Experience Replay
    per_alpha: float = 0.6  # priority exponent
    per_beta_start: float = 0.4  # IS correction initial value
    per_beta_end: float = 1.0  # IS correction final value (annealed)
    per_epsilon: float = 1e-6  # small constant to avoid zero priority

    # Training loop
    num_iterations: int = 10_000
    train_steps_per_iteration: int = 1  # gradient updates per env step

    # Evaluation
    eval_interval: int = 100
    eval_episodes: int = 25

    # Checkpointing / logging
    checkpoint_dir: str | None = os.path.join("models", "rainbow_dqn_checkpoints")
    logs_dir: str = "logs"


# ---------------------------------------------------------------------------
# Transition type
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """A single n-step transition stored in the replay buffer."""

    ideal: tuple  # tuple of np.ndarray, one per polynomial
    selectables: tuple  # tuple of (i, j) pairs
    action: int  # flat action index
    n_step_return: float  # discounted n-step return
    next_ideal: tuple  # next state polynomials
    next_selectables: tuple
    done: bool


# ---------------------------------------------------------------------------
# Sum-tree for PER
# ---------------------------------------------------------------------------


class SumTree:
    """
    Binary sum-tree for O(log N) prioritized sampling.

    Leaves store individual priorities; internal nodes store prefix sums.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data: list[Transition | None] = [None] * capacity
        self._write_pos = 0
        self._size = 0

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        node = leaf_idx
        while node > 1:
            parent = node >> 1
            self._tree[parent] += delta
            node = parent

    def _leaf_index(self, data_index: int) -> int:
        return data_index + self._capacity

    def total_priority(self) -> float:
        return float(self._tree[1])

    def add(self, priority: float, transition: Transition) -> None:
        """Insert a transition with given priority, evicting the oldest."""
        leaf_idx = self._leaf_index(self._write_pos)
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)
        self._data[self._write_pos] = transition
        self._write_pos = (self._write_pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def update_priority(self, data_index: int, priority: float) -> None:
        """Update the priority of the transition at `data_index`."""
        leaf_idx = self._leaf_index(data_index)
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, cumulative_value: float) -> tuple[int, float, "Transition"]:
        """
        Return (data_index, priority, transition) for the leaf whose
        cumulative-priority interval contains `cumulative_value`.

        Clamps cumulative_value to [0, total_priority) to guard against
        floating-point overshoot that would otherwise walk into empty slots.
        """
        cumulative_value = np.clip(cumulative_value, 0.0, self.total_priority() - 1e-6)
        node = 1
        while node < self._capacity:
            left = node << 1
            if cumulative_value <= self._tree[left]:
                node = left
            else:
                cumulative_value -= self._tree[left]
                node = left + 1
        # If the leaf is empty (written position not yet reached), walk left
        # until we find an occupied slot.
        data_index = node - self._capacity
        while self._data[data_index] is None and data_index > 0:
            data_index -= 1
        transition = self._data[data_index]
        assert transition is not None, "SumTree.get called on entirely empty tree"
        return data_index, float(self._tree[data_index + self._capacity]), transition

    def max_priority(self) -> float:
        return float(self._tree[self._capacity : self._capacity + self._size].max())

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """
    Replay buffer backed by a SumTree for O(log N) prioritized sampling.

    New transitions are inserted with maximum current priority so they are
    sampled at least once before receiving a TD-error-based priority.
    """

    _MAX_PRIORITY: float = 1e6

    def __init__(self, capacity: int, alpha: float, epsilon: float):
        self._tree = SumTree(capacity)
        self._alpha = alpha
        self._epsilon = epsilon

    def add(self, transition: Transition, priority: float | None = None) -> None:
        """Add a transition. Uses max priority if `priority` is not given."""
        resolved_priority: float
        if priority is None:
            current_max = self._tree.max_priority() if len(self._tree) > 0 else 1.0
            resolved_priority = min(
                max(current_max, self._epsilon) ** self._alpha, self._MAX_PRIORITY
            )
        else:
            resolved_priority = min(float(priority), self._MAX_PRIORITY)
        self._tree.add(resolved_priority, transition)

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch with importance-sampling weights.

        Returns:
            transitions : list of Transition
            indices     : data indices (for priority updates)
            weights     : IS correction weights, normalised to [0, 1]
        """
        transitions: list[Transition] = []
        indices: list[int] = []
        priorities: list[float] = []

        segment = self._tree.total_priority() / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            # Guard against degenerate range (e.g. near-zero total priority)
            value = np.random.uniform(lo, hi) if hi > lo else lo
            idx, priority, transition = self._tree.get(value)
            transitions.append(transition)
            indices.append(idx)
            priorities.append(max(priority, self._epsilon))

        n = len(self._tree)
        probs = np.array(priorities) / self._tree.total_priority()
        weights = (n * probs) ** (-beta)
        weights /= weights.max()

        return transitions, np.array(indices), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new absolute TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = min(
                (abs(float(td_error)) + self._epsilon) ** self._alpha,
                self._MAX_PRIORITY,
            )
            self._tree.update_priority(int(idx), priority)

    def __len__(self) -> int:
        return len(self._tree)


# ---------------------------------------------------------------------------
# N-step return accumulator
# ---------------------------------------------------------------------------


class NStepBuffer:
    """
    Accumulates raw transitions and emits n-step Transition objects.

    Once the buffer holds n entries (or the episode ends) it emits a
    Transition whose `n_step_return` is the discounted sum of the next n
    rewards and whose next-state is the state n steps ahead.
    """

    def __init__(self, n: int, gamma: float):
        self._n = n
        self._gamma = gamma
        self._buffer: deque[tuple] = deque()

    def add(
        self,
        ideal: tuple,
        selectables: tuple,
        action: int,
        reward: float,
        next_ideal: tuple,
        next_selectables: tuple,
        done: bool,
    ) -> list[Transition]:
        """
        Push one raw (s, a, r, s', done) tuple.

        Returns a (possibly empty) list of ready Transition objects.
        """
        self._buffer.append(
            (ideal, selectables, action, reward, next_ideal, next_selectables, done)
        )
        ready: list[Transition] = []

        if done:
            while self._buffer:
                ready.append(self._build_transition())
                self._buffer.popleft()
        elif len(self._buffer) >= self._n:
            ready.append(self._build_transition())
            self._buffer.popleft()

        return ready

    def _build_transition(self) -> Transition:
        entries = list(self._buffer)
        n_return = 0.0
        for k, entry in enumerate(entries):
            n_return += (self._gamma**k) * entry[3]
            if entry[6]:  # done
                return Transition(
                    ideal=entries[0][0],
                    selectables=entries[0][1],
                    action=entries[0][2],
                    n_step_return=n_return,
                    next_ideal=entry[4],
                    next_selectables=entry[5],
                    done=True,
                )
        last = entries[-1]
        return Transition(
            ideal=entries[0][0],
            selectables=entries[0][1],
            action=entries[0][2],
            n_step_return=n_return,
            next_ideal=last[4],
            next_selectables=last[5],
            done=False,
        )

    def flush(self) -> list[Transition]:
        """Drain all buffered transitions (call at episode end if needed)."""
        ready: list[Transition] = []
        while self._buffer:
            ready.append(self._build_transition())
            self._buffer.popleft()
        return ready


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _obs_to_arrays(
    generators: list, pairs: list
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int, int], ...]]:
    """Convert a raw env observation (PolyElement generators) to storage-friendly numpy tuples.

    Uses make_obs to tokenize PolyElement generators into numeric arrays,
    matching the same representation used during supervised training.
    """
    tokenized_ideal, _ = make_obs(generators, pairs)
    ideal = tuple(np.array(p, dtype=np.float32) for p in tokenized_ideal)
    selectables = tuple(tuple(pair) for pair in pairs)
    return ideal, selectables


def _batch_transitions(
    transitions: list[Transition],
) -> tuple[dict, Array, Array, Array, dict, Array]:
    """
    Collate a list of Transitions into JAX-ready batched arrays.

    Returns:
        obs_batch      : batched current observations (dict)
        actions        : int32 array of flat action indices (B,)
        n_step_returns : float32 n-step returns (B,)
        dones          : float32 terminal flags (B,)
        next_obs_batch : batched next observations (dict)
        loss_mask      : float32, 1.0 for valid samples (B,)
    """
    batch_size = len(transitions)
    max_polys = max(max(len(t.ideal), len(t.next_ideal)) for t in transitions)
    max_monoms = max(max(len(p) for p in t.ideal + t.next_ideal) for t in transitions)
    num_vars = transitions[0].ideal[0].shape[-1]

    def _alloc() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((batch_size, max_polys, max_monoms, num_vars), dtype=np.float32),
            np.zeros((batch_size, max_polys, max_monoms), dtype=bool),
            np.zeros((batch_size, max_polys), dtype=bool),
            np.full((batch_size, max_polys, max_polys), -np.inf, dtype=np.float32),
        )

    ideals, mono_masks, poly_masks, sel_buf = _alloc()
    next_ideals, next_mono_masks, next_poly_masks, next_sel_buf = _alloc()

    actions = np.zeros(batch_size, dtype=np.int32)
    n_step_returns = np.zeros(batch_size, dtype=np.float32)
    dones = np.zeros(batch_size, dtype=np.float32)
    loss_mask = np.ones(batch_size, dtype=np.float32)

    def _fill(
        i: int,
        ideal_tup: tuple,
        sel_tup: tuple,
        buf_ideals: np.ndarray,
        buf_mono_masks: np.ndarray,
        buf_poly_masks: np.ndarray,
        buf_sel: np.ndarray,
    ) -> None:
        buf_poly_masks[i, : len(ideal_tup)] = True
        for j, poly in enumerate(ideal_tup):
            p_len = len(poly)
            buf_ideals[i, j, :p_len] = poly
            buf_mono_masks[i, j, :p_len] = True
        if sel_tup:
            rows, cols = zip(*sel_tup)
            buf_sel[i, rows, cols] = 0.0
        else:
            loss_mask[i] = 0.0

    for i, t in enumerate(transitions):
        _fill(i, t.ideal, t.selectables, ideals, mono_masks, poly_masks, sel_buf)
        _fill(
            i,
            t.next_ideal,
            t.next_selectables,
            next_ideals,
            next_mono_masks,
            next_poly_masks,
            next_sel_buf,
        )
        # t.action was encoded as ai * num_polys_at_step_time + aj.
        # The batched Q-value grid is (max_polys x max_polys), so remap the
        # flat index to use max_polys as the stride instead.
        num_polys_at_step = len(t.ideal)
        ai, aj = t.action // num_polys_at_step, t.action % num_polys_at_step
        actions[i] = ai * max_polys + aj
        n_step_returns[i] = t.n_step_return
        dones[i] = float(t.done)

    def _to_dict(ids, mm, pm, sel) -> dict:
        return {
            "ideals": jnp.array(ids),
            "monomial_masks": jnp.array(mm),
            "poly_masks": jnp.array(pm),
            "selectables": jnp.array(sel),
        }

    return (
        _to_dict(ideals, mono_masks, poly_masks, sel_buf),
        jnp.array(actions),
        jnp.array(n_step_returns),
        jnp.array(dones),
        _to_dict(next_ideals, next_mono_masks, next_poly_masks, next_sel_buf),
        jnp.array(loss_mask),
    )


# ---------------------------------------------------------------------------
# Dueling Q-values
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _rainbow_loss(
    online: GrobnerPolicyValue,
    target: GrobnerPolicyValue,
    obs: dict,
    actions: Array,
    n_step_returns: Array,
    dones: Array,
    next_obs: dict,
    loss_mask: Array,
    is_weights: Array,
    gamma_n: float,
) -> tuple[Array, Array]:
    """
    Double DQN loss with n-step targets and IS-weighted Huber loss.

    Returns:
        weighted_loss : scalar loss for gradient computation.
        td_errors     : per-sample absolute TD errors for priority updates (B,).
    """
    # Double DQN: online selects action, target evaluates it
    next_q_online = eqx.filter_vmap(online.q_values)(next_obs)
    greedy_actions = jnp.argmax(next_q_online, axis=-1)

    next_q_target = eqx.filter_vmap(target.q_values)(next_obs)
    next_q_selected = next_q_target[jnp.arange(actions.shape[0]), greedy_actions]
    next_q_selected = jnp.where(jnp.isfinite(next_q_selected), next_q_selected, 0.0)

    td_target = jnp.clip(
        n_step_returns + gamma_n * next_q_selected * (1.0 - dones),
        a_min=-1e4,
        a_max=1e4,
    )

    q_values = eqx.filter_vmap(online.q_values)(obs)
    q_chosen = q_values[jnp.arange(actions.shape[0]), actions]
    # Guard against NaN/inf that can arise from uninitialised or exploding weights.
    # Masked (padded) entries carry -inf from the selectables mask; replacing them
    # with 0.0 keeps the Huber loss and TD error finite so gradients stay healthy.
    q_chosen = jnp.where(jnp.isfinite(q_chosen), q_chosen, 0.0)

    td_errors = td_target - q_chosen
    per_sample_loss = optax.huber_loss(q_chosen, jax.lax.stop_gradient(td_target))

    weighted_loss = (per_sample_loss * is_weights * loss_mask).sum() / (
        loss_mask.sum() + 1e-9
    )

    return weighted_loss, jnp.abs(td_errors)


# ---------------------------------------------------------------------------
# Gradient step
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _gradient_step(
    online: GrobnerPolicyValue,
    target: GrobnerPolicyValue,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    obs: dict,
    actions: Array,
    n_step_returns: Array,
    dones: Array,
    next_obs: dict,
    loss_mask: Array,
    is_weights: Array,
    gamma_n: float,
) -> tuple[GrobnerPolicyValue, optax.OptState, Array, Array]:
    """Single gradient update of the online network."""

    def loss_fn(m: GrobnerPolicyValue) -> tuple[Array, Array]:
        return _rainbow_loss(
            m,
            target,
            obs,
            actions,
            n_step_returns,
            dones,
            next_obs,
            loss_mask,
            is_weights,
            gamma_n,
        )

    (loss, td_errors), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(online)
    # Clip by global norm before the optimizer step to prevent NaN from
    # exploding gradients early in training.
    grad_arrays = eqx.filter(grads, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(grad_arrays)
    global_norm = jnp.sqrt(jnp.array([jnp.sum(g**2) for g in leaves]).sum())
    clip_factor = jnp.minimum(1.0, 10.0 / (global_norm + 1e-6))
    grads = jax.tree_util.tree_map(
        lambda g: g * clip_factor if eqx.is_array(g) else g, grads
    )
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(online, eqx.is_array)
    )
    updated_online = eqx.apply_updates(online, updates)
    return updated_online, new_opt_state, loss, td_errors


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------


def _polyak_update(
    online: GrobnerPolicyValue,
    target: GrobnerPolicyValue,
    tau: float,
) -> GrobnerPolicyValue:
    """
    Soft (Polyak) parameter update: target = tau * online + (1 - tau) * target.
    tau=1.0 performs a hard copy.
    """
    # eqx.tree_at requires the where-selector to depend only on pytree structure,
    # not leaf values. Use zip over filtered leaves from both models directly.
    return jax.tree_util.tree_map(
        lambda o, t: tau * o + (1.0 - tau) * t if eqx.is_array(o) else t,
        online,
        target,
    )


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


def _select_action_epsilon_greedy(
    online: GrobnerPolicyValue,
    obs_raw: tuple,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Epsilon-greedy action selection restricted to valid (selectable) actions.

    With probability epsilon a uniformly random valid action is taken;
    otherwise the action with the highest dueling Q-value is chosen.
    """
    generators, pairs = obs_raw
    if not pairs:
        return 0

    num_polys = len(generators)

    if rng.random() < epsilon:
        i, j = pairs[rng.integers(len(pairs))]
        return i * num_polys + j

    obs_tokenized = make_obs(generators, pairs)
    q_values = np.array(online.q_values(obs_tokenized))

    mask = np.full(q_values.shape, -np.inf)
    for pi, pj in pairs:
        mask[pi * num_polys + pj] = 0.0
    q_values = q_values + mask

    return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    online: GrobnerPolicyValue,
    env: BuchbergerEnv,
    num_episodes: int,
) -> dict:
    """Greedy evaluation over `num_episodes` episodes."""
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    rng = np.random.default_rng(seed=0)

    for seed in range(num_episodes):
        obs_raw, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = _select_action_epsilon_greedy(
                online, obs_raw, epsilon=0.0, rng=rng
            )
            obs_raw, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }


# ---------------------------------------------------------------------------
# Beta schedule for IS correction
# ---------------------------------------------------------------------------


def _beta_schedule(
    step: int, total_steps: int, beta_start: float, beta_end: float
) -> float:
    """Linearly anneal beta from `beta_start` to `beta_end` over training."""
    fraction = min(step / max(total_steps, 1), 1.0)
    return beta_start + fraction * (beta_end - beta_start)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_rainbow_dqn(
    online: GrobnerPolicyValue,
    env: BuchbergerEnv,
    optimizer: optax.GradientTransformation,
    config: RainbowConfig,
) -> tuple[GrobnerPolicyValue, dict]:
    """
    Rainbow DQN training loop.

    Args:
        online    : Freshly initialised (or pre-trained) online network.
        env       : BuchbergerEnv in 'eval' mode — raw symbolic observations
                    are required for n-step buffering and valid-action masking.
        optimizer : Optax gradient transformation (e.g. optax.nadam).
        config    : Rainbow hyper-parameters.

    Returns:
        Trained online network and a metrics history dict.
    """
    target: GrobnerPolicyValue = jax.tree_util.tree_map(lambda x: x, online)
    opt_state = optimizer.init(eqx.filter(online, eqx.is_array))

    replay_buffer = PrioritizedReplayBuffer(
        capacity=config.replay_buffer_size,
        alpha=config.per_alpha,
        epsilon=config.per_epsilon,
    )
    n_step_buffer = NStepBuffer(n=config.n_step, gamma=config.gamma)
    gamma_n = config.gamma**config.n_step

    rng = np.random.default_rng(seed=42)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = config.num_iterations // 2

    metrics_history: dict[str, list] = {
        "loss": [],
        "mean_reward": [],
        "std_reward": [],
        "mean_length": [],
        "replay_buffer_size": [],
        "epsilon": [],
    }

    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    metrics_log_path = create_metrics_log_path(config.logs_dir)
    best_reward = float("-inf")
    gradient_steps = 0
    episode_seed = 0

    obs_raw, _ = env.reset(seed=episode_seed)

    for iteration in range(config.num_iterations):
        epsilon = max(
            epsilon_end,
            epsilon_start
            - (epsilon_start - epsilon_end) * iteration / epsilon_decay_steps,
        )

        # --- Collect one environment step ---
        action = _select_action_epsilon_greedy(online, obs_raw, epsilon, rng)

        ideal_before, selectables_before = _obs_to_arrays(env.generators, env.pairs)
        num_polys_before = len(env.generators)

        next_obs_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_ideal, next_selectables = _obs_to_arrays(env.generators, env.pairs)

        # Store the action as a flat index within the pre-step grid
        ai = action // num_polys_before
        aj = action % num_polys_before
        flat_action = ai * num_polys_before + aj

        ready_transitions = n_step_buffer.add(
            ideal=ideal_before,
            selectables=selectables_before,
            action=flat_action,
            reward=float(reward),
            next_ideal=next_ideal,
            next_selectables=next_selectables,
            done=done,
        )
        for transition in ready_transitions:
            replay_buffer.add(transition)

        if done:
            episode_seed += 1
            obs_raw, _ = env.reset(seed=episode_seed)
        else:
            obs_raw = next_obs_raw

        # --- Training step (only after warm-up) ---
        iteration_loss: float | None = None
        if len(replay_buffer) >= config.min_replay_size:
            for _ in range(config.train_steps_per_iteration):
                beta = _beta_schedule(
                    gradient_steps,
                    config.num_iterations,
                    config.per_beta_start,
                    config.per_beta_end,
                )
                transitions, indices, is_weights = replay_buffer.sample(
                    config.batch_size, beta
                )
                (
                    obs_batch,
                    actions_batch,
                    returns_batch,
                    dones_batch,
                    next_obs_batch,
                    loss_mask,
                ) = _batch_transitions(transitions)

                online, opt_state, loss, td_errors = _gradient_step(
                    online,
                    target,
                    opt_state,
                    optimizer,
                    obs_batch,
                    actions_batch,
                    returns_batch,
                    dones_batch,
                    next_obs_batch,
                    loss_mask,
                    jnp.array(is_weights),
                    gamma_n,
                )
                replay_buffer.update_priorities(indices, np.array(td_errors))
                gradient_steps += 1
                iteration_loss = float(loss)

                if gradient_steps % config.target_update_interval == 0:
                    target = _polyak_update(online, target, tau=config.tau)

        # --- Periodic evaluation and checkpointing ---
        if (iteration + 1) % config.eval_interval == 0:
            eval_metrics = _evaluate(online, env, config.eval_episodes)
            # _evaluate mutates env state through multiple resets/steps; restore a
            # clean training episode so obs_raw is consistent with the live env.
            obs_raw, _ = env.reset(seed=episode_seed)

            loss_str = f"{iteration_loss:.4f}" if iteration_loss is not None else "n/a"
            print(
                f"Iter {iteration + 1:>6}/{config.num_iterations} | "
                f"eps={epsilon:.3f} | "
                f"buf={len(replay_buffer):>6} | "
                f"loss={loss_str} | "
                f"reward={eval_metrics['mean_reward']:.2f}"
                f" ± {eval_metrics['std_reward']:.2f}"
            )

            metrics_history["mean_reward"].append(eval_metrics["mean_reward"])
            metrics_history["std_reward"].append(eval_metrics["std_reward"])
            metrics_history["mean_length"].append(eval_metrics["mean_length"])

            iteration_metrics: dict = {
                **eval_metrics,
                "epsilon": epsilon,
                "replay_buffer_size": len(replay_buffer),
            }
            if iteration_loss is not None:
                iteration_metrics["loss"] = iteration_loss
                metrics_history["loss"].append(iteration_loss)

            log_metrics(iteration_metrics, metrics_log_path, iteration + 1)

            is_best = eval_metrics["mean_reward"] > best_reward
            if is_best:
                best_reward = eval_metrics["mean_reward"]

            if config.checkpoint_dir:
                save_checkpoint(
                    online,
                    opt_state,
                    config.checkpoint_dir,
                    "last",
                    iteration + 1,
                    iteration_metrics,
                )
                if is_best:
                    save_checkpoint(
                        online,
                        opt_state,
                        config.checkpoint_dir,
                        "best",
                        iteration + 1,
                        iteration_metrics,
                    )
                    print(f"  New best reward: {best_reward:.2f}")

        metrics_history["replay_buffer_size"].append(len(replay_buffer))
        metrics_history["epsilon"].append(epsilon)

    print(f"\nRainbow DQN training complete. Best reward: {best_reward:.2f}")
    return online, metrics_history
