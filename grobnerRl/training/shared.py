"""Shared training utilities for RL algorithms with memory-efficient experience storage."""

from dataclasses import dataclass
import sys
from typing import Dict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue


class PolynomialCache:
    """Cache for deduplicating polynomials with reference counting."""
    
    def __init__(self):
        self._cache: list[np.ndarray] = []
        self._hash_to_idx: Dict[bytes, int] = {}
        self._ref_counts: list[int] = []
        self._free_slots: list[int] = []
    
    def add_polynomial(self, poly: np.ndarray) -> int:
        """Add polynomial and return index. Increments ref count."""
        poly_hash = poly.tobytes()
        
        if poly_hash in self._hash_to_idx:
            idx = self._hash_to_idx[poly_hash]
            self._ref_counts[idx] += 1
            return idx
        
        if self._free_slots:
            idx = self._free_slots.pop()
            self._cache[idx] = poly
            self._ref_counts[idx] = 1
        else:
            idx = len(self._cache)
            self._cache.append(poly)
            self._ref_counts.append(1)
        
        self._hash_to_idx[poly_hash] = idx
        return idx
    
    def remove_reference(self, idx: int) -> None:
        """Decrement ref count. Free slot if count reaches zero."""
        if idx >= len(self._ref_counts):
            return
        
        self._ref_counts[idx] -= 1
        if self._ref_counts[idx] <= 0:
            poly_hash = self._cache[idx].tobytes()
            del self._hash_to_idx[poly_hash]
            self._cache[idx] = None  # type: ignore
            self._free_slots.append(idx)
    
    def get_polynomial(self, idx: int) -> np.ndarray:
        """Get polynomial by index."""
        return self._cache[idx]
    
    def __len__(self) -> int:
        return len(self._cache) - len(self._free_slots)
    
    def memory_usage(self) -> int:
        """Total memory of active polynomials."""
        return sum(poly.nbytes for poly in self._cache if poly is not None)


@dataclass
class Experience:
    """Compressed experience with sparse policy and deduplicated polynomials."""
    
    poly_indices: tuple[int, ...]
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
        poly_cache: PolynomialCache,
    ) -> "Experience":
        """Create compressed experience from uncompressed data."""
        ideal, selectables = observation
        
        poly_indices = tuple(
            poly_cache.add_polynomial(
                poly.astype(np.float32) if poly.dtype != np.float32 else poly
            )
            for poly in ideal
        )
        
        if isinstance(selectables, list):
            selectables = tuple(tuple(pair) for pair in selectables)
        
        nonzero_indices = np.nonzero(policy)[0]
        policy_indices = nonzero_indices.astype(np.int32)
        policy_values = policy[nonzero_indices].astype(np.float32)
        
        return Experience(
            poly_indices=poly_indices,
            selectables=selectables,
            policy_indices=policy_indices,
            policy_values=policy_values,
            value=float(value),
            num_polys=num_polys,
        )
    
    def get_ideal(self, poly_cache: PolynomialCache) -> tuple[np.ndarray, ...]:
        """Reconstruct ideal from polynomial indices."""
        return tuple(poly_cache.get_polynomial(idx) for idx in self.poly_indices)
    
    def observation(self, poly_cache: PolynomialCache) -> tuple:
        """Reconstruct observation tuple."""
        return (self.get_ideal(poly_cache), self.selectables)
    
    @property
    def policy(self) -> np.ndarray:
        """Reconstruct dense policy array."""
        policy = np.zeros(self.num_polys * self.num_polys, dtype=np.float32)
        policy[self.policy_indices] = self.policy_values
        return policy
    
    def memory_usage(self, poly_cache: PolynomialCache, count_polys: bool = True) -> int:
        """Estimate memory usage in bytes."""
        total = sys.getsizeof(self.poly_indices) + len(self.poly_indices) * 8
        
        if count_polys:
            for idx in self.poly_indices:
                total += poly_cache.get_polynomial(idx).nbytes
        
        total += sys.getsizeof(self.selectables)
        total += sum(sys.getsizeof(pair) for pair in self.selectables)
        total += self.policy_indices.nbytes + self.policy_values.nbytes
        total += sys.getsizeof(self.value) + sys.getsizeof(self.num_polys)
        
        return total
    
    def compression_ratio(self, poly_cache: PolynomialCache) -> float:
        """Compression ratio vs uncompressed format."""
        compressed = self.memory_usage(poly_cache, count_polys=True)
        
        uncompressed = sum(
            poly_cache.get_polynomial(idx).nbytes for idx in self.poly_indices
        )
        uncompressed += sys.getsizeof(self.selectables)
        uncompressed += sum(sys.getsizeof(pair) for pair in self.selectables)
        uncompressed += (self.num_polys ** 2) * 4
        uncompressed += sys.getsizeof(self.value) + sys.getsizeof(self.num_polys)
        
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
    """Replay buffer with polynomial deduplication and automatic cleanup."""

    def __init__(self, max_size: int = 100000, poly_cache: PolynomialCache | None = None):
        self.max_size = max_size
        self.buffer: list[Experience] = []
        self.position = 0
        self.poly_cache = poly_cache if poly_cache is not None else PolynomialCache()

    def add(self, experiences: list[Experience]) -> None:
        """Add experiences to buffer. Cleans up old experiences when full."""
        for exp in experiences:
            if len(self.buffer) < self.max_size:
                self.buffer.append(exp)
            else:
                old_exp = self.buffer[self.position]
                for idx in old_exp.poly_indices:
                    self.poly_cache.remove_reference(idx)
                self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch of experiences."""
        indices = np.random.choice(
            len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False
        )
        return [self.buffer[i] for i in indices]
    
    def sample_batched(self, batch_size: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Sample and batch experiences."""
        return batch_experiences(self.sample(batch_size), self.poly_cache)

    def __len__(self) -> int:
        return len(self.buffer)
    
    def memory_usage(self) -> dict:
        """Memory usage statistics including deduplication savings."""
        if len(self.buffer) == 0:
            return {
                "total_bytes": 0,
                "total_mb": 0.0,
                "avg_bytes_per_experience": 0,
                "avg_compression_ratio": 1.0,
                "poly_cache_bytes": 0,
                "poly_cache_mb": 0.0,
                "unique_polynomials": 0,
                "deduplication_ratio": 1.0,
            }
        
        exp_bytes = sum(exp.memory_usage(self.poly_cache, count_polys=False) 
                       for exp in self.buffer)
        cache_bytes = self.poly_cache.memory_usage()
        total_bytes = exp_bytes + cache_bytes
        
        total_poly_refs = sum(len(exp.poly_indices) for exp in self.buffer)
        num_unique = len(self.poly_cache)
        dedup_ratio = num_unique / total_poly_refs if total_poly_refs > 0 else 1.0
        
        avg_compression = np.mean([exp.compression_ratio(self.poly_cache) 
                                   for exp in self.buffer])
        
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "avg_bytes_per_experience": total_bytes / len(self.buffer),
            "avg_compression_ratio": float(avg_compression),
            "poly_cache_bytes": cache_bytes,
            "poly_cache_mb": cache_bytes / (1024 * 1024),
            "unique_polynomials": num_unique,
            "deduplication_ratio": float(dedup_ratio),
        }


def batch_experiences(
    experiences: list[Experience],
    poly_cache: PolynomialCache,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Batch compressed experiences for training."""
    batch_size = len(experiences)
    max_polys = max(exp.num_polys for exp in experiences)
    
    first_ideal = experiences[0].get_ideal(poly_cache)
    max_monoms = max(
        max(len(p) for p in exp.get_ideal(poly_cache)) for exp in experiences
    )
    num_vars = len(first_ideal[0][0])

    batched_ideals = np.zeros(
        (batch_size, max_polys, max_monoms, num_vars), dtype=np.float32
    )
    batched_monomial_masks = np.zeros((batch_size, max_polys, max_monoms), dtype=bool)
    batched_poly_masks = np.zeros((batch_size, max_polys), dtype=bool)
    batched_selectables = np.full(
        (batch_size, max_polys, max_polys), -np.inf, dtype=np.float32
    )
    batched_policies = np.zeros((batch_size, max_polys * max_polys), dtype=np.float32)
    batched_values = np.zeros(batch_size, dtype=np.float32)
    loss_mask = np.ones(batch_size, dtype=np.float32)

    for i, exp in enumerate(experiences):
        ideal = exp.get_ideal(poly_cache)
        batched_poly_masks[i, :len(ideal)] = True

        for j, poly in enumerate(ideal):
            p_len = len(poly)
            batched_ideals[i, j, :p_len] = poly
            batched_monomial_masks[i, j, :p_len] = True

        if exp.selectables:
            rows, cols = zip(*exp.selectables)
            batched_selectables[i, rows, cols] = 0.0

        for idx, val in zip(exp.policy_indices, exp.policy_values):
            orig_i = idx // exp.num_polys
            orig_j = idx % exp.num_polys
            batched_policies[i, orig_i * max_polys + orig_j] = val

        batched_values[i] = exp.value

    return {
        "ideals": batched_ideals,
        "monomial_masks": batched_monomial_masks,
        "poly_masks": batched_poly_masks,
        "selectables": batched_selectables,
    }, batched_policies, batched_values, loss_mask


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
