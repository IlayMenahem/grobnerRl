from dataclasses import dataclass
from collections import deque
import random
import jax.numpy as jnp

from .utils import GroebnerState


@dataclass(frozen=True)
class TimeStep:
    obs: GroebnerState
    action: tuple[int, ...] | int
    reward: float
    next_obs: GroebnerState
    done: bool


class ReplayBuffer:
    queue: deque[TimeStep]
    max_size: int
    batch_size: int

    def __init__(self, size: int, batch_size: int) -> None:
        self.queue = deque(maxlen=size)
        self.max_size = size
        self.batch_size = batch_size

    def store(self, obs: GroebnerState, act: tuple[int, ...] | int, rew: float, next_obs: GroebnerState, done: bool) -> None:
        self.queue.append(TimeStep(obs, act, rew, next_obs, done))

    def sample_batch(self) -> dict[str, jnp.ndarray]:
        indecies = random.sample(range(len(self.queue)), k=self.batch_size)
        samples = [self.queue[i] for i in indecies]

        batch = {'obs': [t.obs for t in samples],
                'next_obs': [t.next_obs for t in samples],
                'acts': [t.action for t in samples],
                'rews': [t.reward for t in samples],
                'done': [t.done for t in samples]}

        return batch

    def can_sample(self) -> bool:
        return len(self.queue) >= self.batch_size
