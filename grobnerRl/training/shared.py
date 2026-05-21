"""Shared training utilities for RL algorithms with Grain-based experience storage."""

import json
import os
import tempfile
from copy import copy
from dataclasses import dataclass
from typing import Iterator

import equinox as eqx
import grain.python as grain
import jax.numpy as jnp
import numpy as np
import optax
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.transforms import Batch
from jaxtyping import Array

from grobnerRl.env import BuchbergerEnv, make_obs
from grobnerRl.models import GrobnerPolicyValue


class MinMaxStats:
    """
    Tracks min/max values across a search tree for normalisation.

    Values are normalised to [0, 1] using the global min/max seen during
    tree search, ensuring that exploration bonuses operate on a consistent
    scale regardless of the environment's reward magnitude.
    """

    def __init__(self) -> None:
        self.maximum: float = -float("inf")
        self.minimum: float = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def copy_env(env: BuchbergerEnv) -> BuchbergerEnv:
    """Create a shallow copy of the environment with independent generator and pair lists."""
    new_env = copy(env)
    new_env.generators = list(env.generators)
    new_env.pairs = list(env.pairs)
    return new_env


def get_valid_actions(env: BuchbergerEnv) -> list[int]:
    """Return valid actions as flattened pair indices for the current environment state."""
    num_polys = len(env.generators)
    return [i * num_polys + j for i, j in env.pairs]


@dataclass
class Experience:
    """Uncompressed experience for Grain-based replay buffer."""

    ideal: tuple[np.ndarray, ...]
    selectables: tuple[tuple[int, int], ...]
    policy: np.ndarray  # Dense policy array over (num_polys * num_polys) actions
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
    """Grain data source for experiences stored in a JSON file."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.experiences: list[Experience] = []

        if os.path.exists(storage_path):
            with open(storage_path, "r") as f:
                data = json.load(f)
                self.experiences = [
                    _deserialize_experience(exp) for exp in data.get("experiences", [])
                ]

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, index) -> Experience:
        if isinstance(index, slice):
            raise TypeError("Slicing not supported, use individual indices")
        return self.experiences[int(index)]


class GrainReplayBuffer:
    """
    Grain-based replay buffer using file-backed circular storage.

    FIFO eviction is driven purely by the on-disk count so that position and
    fullness state survive process restarts without any extra bookkeeping file.
    """

    def __init__(
        self,
        max_size: int = 100000,
        storage_dir: str | None = None,
        worker_count: int = 1,
        worker_buffer_size: int = 4,
    ):
        self.max_size = max_size
        self.worker_count = worker_count
        self.worker_buffer_size = worker_buffer_size

        if storage_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="grain_replay_")
            self.storage_dir = self._temp_dir
        else:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
            self._temp_dir = None

        self.storage_path = os.path.join(self.storage_dir, "experiences.json")

        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w") as f:
                json.dump({"experiences": []}, f)

        self._data_source = ExperienceDataSource(self.storage_path)
        self._current_epoch = 0

        # Derive circular-buffer state from the existing on-disk count so that
        # resuming a run never appends past max_size.
        existing_count = len(self._data_source)
        self._is_full: bool = existing_count >= self.max_size
        self._position: int = (
            existing_count % self.max_size if self._is_full else existing_count
        )

    def add(self, experiences: list[Experience]) -> None:
        """Add experiences to the buffer with FIFO eviction once full."""
        with open(self.storage_path, "r") as f:
            stored_exps: list[dict] = json.load(f)["experiences"]

        for exp in experiences:
            serialized = _serialize_experience(exp)
            if self._is_full:
                stored_exps[self._position] = serialized
            else:
                stored_exps.append(serialized)
                if len(stored_exps) >= self.max_size:
                    self._is_full = True

            self._position = (self._position + 1) % self.max_size

        with open(self.storage_path, "w") as f:
            json.dump({"experiences": stored_exps}, f)

        self._data_source.experiences = [
            _deserialize_experience(exp) for exp in stored_exps
        ]

    def _create_dataloader(
        self, batch_size: int, shuffle: bool = True
    ) -> DataLoader | None:
        """Create a Grain DataLoader over the current buffer contents."""
        if len(self._data_source) == 0:
            return None

        data_source = ExperienceDataSource(self.storage_path)

        sampler = IndexSampler(
            len(data_source),
            shard_options=ShardOptions(0, 1, True),
            shuffle=shuffle,
            num_epochs=1,
            seed=self._current_epoch if shuffle else 0,
        )

        batch_transform = Batch(
            batch_size,
            drop_remainder=False,
            batch_fn=lambda exps: batch_experiences(list(exps)),
        )

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

    def sample_batched(
        self, batch_size: int
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Sample one batch of experiences."""
        for batch in self.iter_dataset(batch_size=batch_size, shuffle=True):
            return batch
        return {}, np.array([]), np.array([]), np.array([])

    def iter_dataset(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over all experiences in batches using a Grain DataLoader."""
        dataloader = self._create_dataloader(batch_size=batch_size, shuffle=shuffle)
        return iter(dataloader) if dataloader is not None else iter([])

    def __len__(self) -> int:
        return len(self._data_source)

    def __del__(self):
        if self._temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass


def batch_experiences(
    experiences: list[Experience],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Batch a list of experiences into padded arrays ready for training."""
    if not experiences:
        return {}, np.array([]), np.array([]), np.array([])

    batch_size = len(experiences)
    max_polys = max(exp.num_polys for exp in experiences)
    max_monoms = max(len(poly) for exp in experiences for poly in exp.ideal)
    num_vars = len(experiences[0].ideal[0][0])

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
        batched_poly_masks[i, : len(exp.ideal)] = True

        for j, poly in enumerate(exp.ideal):
            batched_ideals[i, j, : len(poly)] = poly
            batched_monomial_masks[i, j, : len(poly)] = True

        if exp.selectables:
            rows, cols = zip(*exp.selectables)
            batched_selectables[i, rows, cols] = 0.0

        for idx, prob in enumerate(exp.policy):
            if prob > 0:
                orig_i, orig_j = idx // exp.num_polys, idx % exp.num_polys
                batched_policies[i, orig_i * max_polys + orig_j] = prob

        batched_values[i] = exp.value

    observations = {
        "ideals": batched_ideals,
        "monomial_masks": batched_monomial_masks,
        "poly_masks": batched_poly_masks,
        "selectables": batched_selectables,
    }
    return observations, batched_policies, batched_values, loss_mask


def policy_value_loss(
    model: GrobnerPolicyValue,
    observations: dict,
    target_policies: Array,
    values: Array,
    loss_mask: Array,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> tuple[Array, dict]:
    """
    Compute combined policy and value loss.

    Args:
        model: The GrobnerPolicyValue model.
        observations: Batched observations dict.
        target_policies: Target policy distributions (batch_size, max_actions).
        values: Target state values (batch_size,).
        loss_mask: Binary mask for valid samples (batch_size,).
        policy_loss_weight: Scalar weight applied to the policy loss term.
        value_loss_weight: Scalar weight applied to the value loss term.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    policy_logits, pred_values = eqx.filter_vmap(model)(observations)

    policy_loss = optax.safe_softmax_cross_entropy(policy_logits, target_policies)
    value_loss = optax.huber_loss(pred_values, values)
    total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss

    normalizer = loss_mask.sum() + 1e-9
    masked_total_loss = (total_loss * loss_mask).sum() / normalizer
    masked_policy_loss = (policy_loss * loss_mask).sum() / normalizer
    masked_value_loss = (value_loss * loss_mask).sum() / normalizer

    metrics = {
        "policy_loss": masked_policy_loss,
        "value_loss": masked_value_loss,
        "total_loss": masked_total_loss,
    }
    return masked_total_loss, metrics


def train_policy_value(
    model: GrobnerPolicyValue,
    replay_buffer: GrainReplayBuffer,
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
        Tuple of (trained_model, new_opt_state, mean_metrics).
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
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: policy_value_loss(
                m,
                observations,
                target_policies,
                values,
                loss_mask,
                train_config.policy_loss_weight,
                train_config.value_loss_weight,
            ),
            has_aux=True,
        )(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss, metrics

    epoch_metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
    }

    for _ in range(train_config.num_epochs_per_iteration):
        for (
            observations,
            target_policies,
            values,
            loss_mask,
        ) in replay_buffer.iter_dataset(
            batch_size=train_config.batch_size, shuffle=True
        ):
            observations = {k: jnp.array(v) for k, v in observations.items()}
            target_policies = jnp.array(target_policies)
            values = jnp.array(values)
            loss_mask = jnp.array(loss_mask)

            model, opt_state, _, metrics = make_step(
                model, opt_state, observations, target_policies, values, loss_mask
            )

            for k, v in metrics.items():
                epoch_metrics[k].append(float(v))

    mean_metrics = {
        k: float(np.mean(v)) if v else 0.0 for k, v in epoch_metrics.items()
    }
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
        Dictionary with mean/std of reward and episode length.
    """

    def greedy_action(
        policy_logits: np.ndarray, pairs: list[tuple[int, int]], num_polys: int
    ) -> tuple[int, int]:
        mask = np.full(policy_logits.shape, float("-inf"))
        for i, j in pairs:
            mask[i * num_polys + j] = 0.0
        action = int(np.argmax(policy_logits + mask))
        return action // num_polys, action % num_polys

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for seed in range(num_episodes):
        env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            policy_logits, _ = model(make_obs(env.generators, env.pairs))
            i, j = greedy_action(
                np.array(policy_logits), env.pairs, len(env.generators)
            )
            _, reward, terminated, truncated, _ = env.step((i, j))
            total_reward += reward
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
