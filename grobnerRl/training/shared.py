"""
Shared training utilities for RL algorithms (AlphaZero, Gumbel MuZero).

This module provides common components used across different RL training
algorithms, including experience storage, batching, loss functions, and
training loops.
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue


@dataclass
class Experience:
    """A single experience from self-play."""

    observation: tuple
    policy: np.ndarray
    value: float
    num_polys: int


@dataclass
class TrainConfig:
    """Configuration for RL training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs_per_iteration: int = 3
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


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


def policy_value_loss(
    model: GrobnerPolicyValue,
    observations: dict,
    target_policies: Array,
    values: Array,
    loss_mask: Array,
) -> tuple[Array, dict]:
    """
    Compute combined policy and value loss.

    Args:
        model: The GrobnerPolicyValue model.
        observations: Batched observations dict.
        target_policies: Target policies (batch_size, max_actions).
        values: Target values (batch_size,).
        loss_mask: Mask for valid samples (batch_size,).

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    policy_logits, pred_values = eqx.filter_vmap(model)(observations)

    policy_loss = optax.softmax_cross_entropy(policy_logits, target_policies, where=target_policies > 0)
    value_loss = optax.huber_loss(pred_values, values)

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


def train_policy_value(
    model: GrobnerPolicyValue,
    replay_buffer: ReplayBuffer,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[GrobnerPolicyValue, optax.OptState, dict]:
    """
    Train the model on replay buffer data.

    Args:
        model: The GrobnerPolicyValue model.
        replay_buffer: Buffer containing self-play experiences.
        train_config: Training configuration.
        optimizer: Optax optimizer.
        opt_state: Current optimizer state.

    Returns:
        Tuple of (trained_model, new_opt_state, metrics).
    """

    @eqx.filter_jit
    def make_step(
        model: GrobnerPolicyValue,
        opt_state: optax.OptState,
        observations: dict,
        target_policies: Array,
        values: Array,
        loss_mask: Array,
    ) -> tuple[GrobnerPolicyValue, optax.OptState, Array, dict]:
        def loss_fn(m):
            return policy_value_loss(m, observations, target_policies, values, loss_mask)

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
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    num_episodes: int = 20,
) -> dict:
    """
    Evaluate the model by playing episodes greedily.

    Args:
        model: The GrobnerPolicyValue model.
        env: Environment for evaluation.
        num_episodes: Number of episodes to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
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
