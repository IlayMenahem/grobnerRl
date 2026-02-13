"""
Shared training utilities for RL algorithms (AlphaZero, Gumbel MuZero).

This module provides common components used across different RL training
algorithms, including experience storage, batching, loss functions, and
training loops.
"""

from dataclasses import dataclass
import sys

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue


@dataclass
class Experience:
    """
    A compressed experience from self-play with lossless compression.
    
    Memory optimization:
    - Policy stored sparsely (only non-zero entries)
    - Observation stored with efficient dtypes
    """
    ideal: tuple[np.ndarray, ...]
    selectables: tuple[tuple[int, int], ...]
    
    policy_indices: np.ndarray
    policy_values: np.ndarray
    
    value: float
    num_polys: int
    
    @staticmethod
    def from_uncompressed(
        observation: tuple,
        policy: np.ndarray,
        value: float,
        num_polys: int,
    ) -> "Experience":
        """Create compressed experience from uncompressed data."""
        ideal, selectables = observation
        
        # Ensure ideal polynomials are float32 (not float64)
        ideal_compressed = tuple(
            poly.astype(np.float32) if poly.dtype != np.float32 else poly
            for poly in ideal
        )
        
        # Convert selectables to tuple of tuples (immutable, memory efficient)
        if isinstance(selectables, list):
            selectables_compressed = tuple(tuple(pair) for pair in selectables)
        else:
            selectables_compressed = selectables
        
        # Compress policy (sparse representation)
        nonzero_indices = np.nonzero(policy)[0]
        policy_indices = nonzero_indices.astype(np.int32)
        policy_values = policy[nonzero_indices].astype(np.float32)
        
        return Experience(
            ideal=ideal_compressed,
            selectables=selectables_compressed,
            policy_indices=policy_indices,
            policy_values=policy_values,
            value=float(value),
            num_polys=num_polys,
        )
    
    @property
    def observation(self) -> tuple:
        """Reconstruct observation tuple for backward compatibility."""
        return (self.ideal, self.selectables)
    
    @property
    def policy(self) -> np.ndarray:
        """Reconstruct dense policy array."""
        policy = np.zeros(self.num_polys * self.num_polys, dtype=np.float32)
        policy[self.policy_indices] = self.policy_values
        return policy
    
    def memory_usage(self) -> int:
        """
        Estimate memory usage in bytes.
        
        Returns:
            Approximate memory usage in bytes.
        """
        total = 0
        
        # Ideal polynomials
        for poly in self.ideal:
            total += poly.nbytes
        
        # Selectables (tuple of tuples)
        total += sys.getsizeof(self.selectables)
        total += sum(sys.getsizeof(pair) for pair in self.selectables)
        
        # Sparse policy
        total += self.policy_indices.nbytes
        total += self.policy_values.nbytes
        
        # Scalars
        total += sys.getsizeof(self.value)
        total += sys.getsizeof(self.num_polys)
        
        return total
    
    def compression_ratio(self) -> float:
        """
        Compute compression ratio vs uncompressed format.
        
        Returns:
            Ratio of compressed size to uncompressed size.
        """
        # Compressed size
        compressed = self.memory_usage()
        
        # Uncompressed size (dense policy + observation)
        uncompressed = 0
        for poly in self.ideal:
            uncompressed += poly.nbytes
        uncompressed += sys.getsizeof(self.selectables)
        uncompressed += sum(sys.getsizeof(pair) for pair in self.selectables)
        uncompressed += (self.num_polys ** 2) * 4  # float32 dense policy
        uncompressed += sys.getsizeof(self.value)
        uncompressed += sys.getsizeof(self.num_polys)
        
        return compressed / uncompressed if uncompressed > 0 else 1.0


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
    
    def sample_batched(self, batch_size: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences and batch them."""
        experiences = self.sample(batch_size)
        return batch_experiences(experiences)

    def __len__(self) -> int:
        return len(self.buffer)
    
    def memory_usage(self) -> dict:
        """
        Compute memory usage statistics for the replay buffer.
        
        Returns:
            Dictionary with memory statistics in bytes and MB.
        """
        if len(self.buffer) == 0:
            return {
                "total_bytes": 0,
                "total_mb": 0.0,
                "avg_bytes_per_experience": 0,
                "avg_compression_ratio": 1.0,
            }
        
        total_bytes = sum(exp.memory_usage() for exp in self.buffer)
        avg_compression = np.mean([exp.compression_ratio() for exp in self.buffer])
        
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "avg_bytes_per_experience": total_bytes / len(self.buffer),
            "avg_compression_ratio": float(avg_compression),
        }


def batch_experiences(
    experiences: list[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch a list of compressed experiences for training.
    
    Works directly with compressed format to avoid unnecessary decompression.
    """
    batch_size = len(experiences)

    max_polys = max(exp.num_polys for exp in experiences)
    max_monoms = max(
        max(len(p) for p in exp.ideal) for exp in experiences
    )
    num_vars = len(experiences[0].ideal[0][0])

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
        ideal = exp.ideal
        selectables = exp.selectables
        num_polys = len(ideal)

        batched_poly_masks[i, :num_polys] = True

        for j, poly in enumerate(ideal):
            p_len = len(poly)
            batched_ideals[i, j, :p_len] = poly
            batched_monomial_masks[i, j, :p_len] = True

        if selectables:
            rows, cols = zip(*selectables)
            batched_selectables[i, rows, cols] = 0.0

        original_num_polys = exp.num_polys
        for idx, val in zip(exp.policy_indices, exp.policy_values):
            # Map from original action space to batched action space
            orig_i = idx // original_num_polys
            orig_j = idx % original_num_polys
            new_action = orig_i * max_polys + orig_j
            batched_policies[i, new_action] = val

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
            observations, target_policies, values, loss_mask = replay_buffer.sample_batched(train_config.batch_size)

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
