import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import optax
import rlax
import gymnasium as gym
from collections import deque
import random


class poll_agent(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, input_size: int, output_size: int, key):
        self.linear1 = eqx.nn.Linear(input_size, input_size, key=key)
        self.linear2 = eqx.nn.Linear(input_size, output_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self) -> tuple[jnp.ndarray, ...]:
        data = zip(*random.sample(self.buffer, self.batch_size))
        data = tuple(jnp.array(x) for x in data)

        return data

    def is_ready(self):
        return len(self.buffer) >= self.batch_size

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, gamma, q_network, num_episodes, epsilon_scheduler, optimizer, replay_buffer, target_update_freq):
    '''
    trains a DQN agent

    Args:
    - env (gym.Env): gym environment
    - gamma (int): discount factor
    - q_network (eqx.nn.Module): DQN agent
    - num episodes (int): number of episodes to train for
    - epsilon_scheduler: epsilon scheduler for epsilon-greedy exploration
    - optimizer: optimizer for training
    - replay_buffer: replay buffer for experience replay
    - target_update_freq (int): frequency of target network updates

    returns:
    - q_network: trained DQN agent
    '''
    target_network = eqx.nn.inference_mode(q_network)

    for episode in range(num_episodes):
        pass

    raise NotImplementedError

def train_rainbow():
    raise NotImplementedError

def train_ppo():
    raise NotImplementedError

def train_a3c():
    raise NotImplementedError

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    num_episodes = 2000
    gamma = 0.99
    seed = 0
    target_update_freq = 50

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    q_network = poll_agent(input_size=4, output_size=2, key=subkey)

    capacity = 2500
    batch_size = 256
    replay_buffer = ReplayBuffer(capacity, batch_size)

    epsilon_shed = optax.schedules.exponential_decay(1.0, 1, 0.995, end_value=0.05)
    optimizer = optax.adam(1e-4)

    train_dqn()
