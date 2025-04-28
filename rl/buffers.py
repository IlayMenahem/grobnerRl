from dataclasses import dataclass
from collections import deque
import random
import jax.numpy as jnp

@dataclass(frozen=True)
class TimeStep:
    obs: jnp.ndarray
    action: int
    reward: float
    next_obs: jnp.ndarray
    done: bool


class ReplayBuffer:
    queue: deque[TimeStep]
    max_size: int
    batch_size: int

    def __init__(self, size: int, batch_size: int) -> None:
        self.queue = deque(maxlen=size)
        self.max_size = size
        self.batch_size = batch_size

    def store(self, obs: jnp.ndarray, act: int, rew: float, next_obs: jnp.ndarray, done: bool) -> None:
        self.queue.append(TimeStep(obs, act, rew, next_obs, done))

    def sample_batch(self) -> dict[str, jnp.ndarray]:
        indecies = random.sample(range(len(self.queue)), k=self.batch_size)
        samples = [self.queue[i] for i in indecies]
        batch = {'obs': jnp.array([t.obs for t in samples]),
                'next_obs': jnp.array([t.next_obs for t in samples]),
                'acts': jnp.array([t.action for t in samples]),
                'rews': jnp.array([t.reward for t in samples]),
                'done': jnp.array([t.done for t in samples])}

        return batch

    def can_sample(self) -> bool:
        return len(self.queue) >= self.batch_size
