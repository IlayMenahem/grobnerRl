"""Shared training utilities for RL algorithms with Grain-based experience storage."""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterator

import equinox as eqx
import grain.python as grain
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.transforms import Batch
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue


@dataclass
class Experience:
    """Uncompressed experience for Grain-based replay buffer."""
    
    ideal: tuple[np.ndarray, ...]  # Raw polynomials
    selectables: tuple[tuple[int, int], ...]
    policy: np.ndarray  # Dense policy array
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
    worker_count: int = 1
    worker_buffer_size: int = 4


def _serialize_experience(exp: Experience) -> dict:
    """Serialize an Experience to a JSON-compatible dict."""
    return {
        "ideal": [poly.tolist() for poly in exp.ideal],
        "selectables": list(exp.selectables),
        "policy": exp.policy.tolist(),
        "value": float(exp.value),
        "num_polys": int(exp.num_polys),
    }


def _deserialize_experience(data: dict) -> Experience:
    """Deserialize an Experience from a JSON dict."""
    return Experience(
        ideal=tuple(np.array(poly, dtype=np.float32) for poly in data["ideal"]),
        selectables=tuple(tuple(pair) for pair in data["selectables"]),
        policy=np.array(data["policy"], dtype=np.float32),
        value=float(data["value"]),
        num_polys=int(data["num_polys"]),
    )


class ExperienceDataSource(grain.RandomAccessDataSource):
    """Grain data source for experiences stored in JSON file."""
    
    def __init__(self, storage_path: str):
        """
        Initialize data source from storage file.
        
        Args:
            storage_path: Path to JSON file containing experiences
        """
        self.storage_path = storage_path
        self.experiences = []
        
        # Load and deserialize all experiences if file exists
        if os.path.exists(storage_path):
            with open(storage_path, "r") as f:
                data = json.load(f)
                raw_experiences = data.get("experiences", [])
                self.experiences = [_deserialize_experience(exp) for exp in raw_experiences]
    
    def __len__(self) -> int:
        return len(self.experiences)
    
    def __getitem__(self, index) -> Experience:
        if isinstance(index, slice):
            raise TypeError("Slicing not supported, use individual indices")
        
        return self.experiences[int(index)]


class GrainReplayBuffer:
    """Grain-based replay buffer using file storage and IndexSampler."""

    def __init__(
        self,
        max_size: int = 100000,
        storage_dir: str | None = None,
        worker_count: int = 1,
        worker_buffer_size: int = 4,
    ):
        """
        Initialize replay buffer with file-based storage.
        
        Args:
            max_size: Maximum number of experiences to store
            storage_dir: Directory to store experience files (uses temp dir if None)
            worker_count: Number of workers for data loading
            worker_buffer_size: Buffer size per worker
        """
        self.max_size = max_size
        self.worker_count = worker_count
        self.worker_buffer_size = worker_buffer_size
        
        # Setup storage
        if storage_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="grain_replay_")
            self.storage_dir = self._temp_dir
        else:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
            self._temp_dir = None
        
        self.storage_path = os.path.join(self.storage_dir, "experiences.json")
        
        # Initialize with empty file
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w") as f:
                json.dump({"experiences": []}, f)
        
        self._data_source = ExperienceDataSource(self.storage_path)
        self._current_epoch = 0
        self._experiences_buffer: list[dict] = []
        
        # Track FIFO position for circular buffer
        self._position = 0
        self._is_full = False

    def add(self, experiences: list[Experience]) -> None:
        """Add experiences to buffer with FIFO eviction when full."""
        # Load existing experiences
        with open(self.storage_path, "r") as f:
            data = json.load(f)
            stored_exps = data["experiences"]
        
        # Add new experiences with FIFO eviction
        for exp in experiences:
            serialized = _serialize_experience(exp)
            
            if not self._is_full and len(stored_exps) < self.max_size:
                stored_exps.append(serialized)
            else:
                # FIFO: replace oldest experience
                self._is_full = True
                stored_exps[self._position] = serialized
                self._position = (self._position + 1) % self.max_size
        
        # Write back to file
        with open(self.storage_path, "w") as f:
            json.dump({"experiences": stored_exps}, f)
        
        # Update data source with deserialized experiences
        self._data_source.experiences = [_deserialize_experience(exp) for exp in stored_exps]

    def _create_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader | None:
        """Create Grain DataLoader with IndexSampler and multiple workers."""
        if len(self._data_source) == 0:
            return None
        
        # Recreate data source to get fresh file handle
        data_source = ExperienceDataSource(self.storage_path)
        
        # Create IndexSampler with current epoch as seed
        sampler = IndexSampler(
            len(data_source),
            shard_options=ShardOptions(0, 1, True),
            shuffle=shuffle,
            num_epochs=1,
            seed=self._current_epoch if shuffle else 0,
        )
        
        # Batch transformation
        batch_transform = Batch(batch_size, drop_remainder=False, batch_fn=batch_experiences_grain)
        
        # Create DataLoader with workers
        dataloader = DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=(batch_transform,),
            worker_count=self.worker_count,
            worker_buffer_size=self.worker_buffer_size,
        )
        
        if shuffle:
            self._current_epoch += 1
        
        return dataloader

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch of experiences."""
        if len(self._data_source) == 0:
            return []
        
        indices = np.random.choice(
            len(self._data_source), 
            size=min(batch_size, len(self._data_source)), 
            replace=False
        )
        return [self._data_source[int(i)] for i in indices]
    
    def sample_batched(self, batch_size: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Sample and batch experiences using Grain DataLoader with IndexSampler."""
        dataloader = self._create_dataloader(batch_size=batch_size, shuffle=True)
        
        if dataloader is None:
            return {}, np.array([]), np.array([]), np.array([])
        
        # Get first batch from dataloader
        for batch in dataloader:
            return batch
        
        # If no batch available, return empty
        return {}, np.array([]), np.array([]), np.array([])
    
    def iter_dataset(self, batch_size: int, shuffle: bool = True) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create iterator over batches using Grain DataLoader.
        
        Useful for multiple epochs of training on the same data.
        """
        dataloader = self._create_dataloader(batch_size=batch_size, shuffle=shuffle)
        
        if dataloader is None:
            return iter([])
        
        return iter(dataloader)

    def __len__(self) -> int:
        return len(self._data_source)
    
    def __del__(self):
        """Cleanup temporary directory if created."""
        if self._temp_dir is not None:
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass


def batch_experiences_grain(
    experiences: Sequence[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch function for Grain DataLoader.
    
    Compatible with Grain's Batch transform.
    """
    return batch_experiences(list(experiences))


def batch_experiences(
    experiences: list[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Batch uncompressed experiences for training."""
    if not experiences:
        return {}, np.array([]), np.array([]), np.array([])
    
    batch_size = len(experiences)
    max_polys = max(exp.num_polys for exp in experiences)
    
    first_ideal = experiences[0].ideal
    max_monoms = max(
        max(len(p) for p in exp.ideal) for exp in experiences
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
        ideal = exp.ideal
        batched_poly_masks[i, :len(ideal)] = True

        for j, poly in enumerate(ideal):
            p_len = len(poly)
            batched_ideals[i, j, :p_len] = poly
            batched_monomial_masks[i, j, :p_len] = True

        if exp.selectables:
            rows, cols = zip(*exp.selectables)
            batched_selectables[i, rows, cols] = 0.0

        for idx in range(len(exp.policy)):
            if exp.policy[idx] > 0:
                orig_i = idx // exp.num_polys
                orig_j = idx % exp.num_polys
                batched_policies[i, orig_i * max_polys + orig_j] = exp.policy[idx]

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
    replay_buffer: GrainReplayBuffer,
    train_config: TrainConfig,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[GrobnerPolicyValue, optax.OptState, dict]:
    """
    Train the model on replay buffer data using Grain DataLoader.

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

    for _ in range(train_config.num_epochs_per_iteration):
        # Use iter_dataset to iterate through all batches
        for observations, target_policies, values, loss_mask in replay_buffer.iter_dataset(
            batch_size=train_config.batch_size, shuffle=True
        ):
            # Convert to JAX arrays
            observations = {k: jnp.array(v) for k, v in observations.items()}
            target_policies = jnp.array(target_policies)
            values = jnp.array(values)
            loss_mask = jnp.array(loss_mask)

            model, opt_state, _, metrics = make_step(
                model, opt_state, observations, target_policies, values, loss_mask
            )

            for k, v in metrics.items():
                epoch_metrics[k].append(float(v))

    mean_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
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
