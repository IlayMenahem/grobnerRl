from dataclasses import dataclass
from collections import deque
import random

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import gymnasium as gym


class poll_agent(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    state_value: eqx.nn.Linear
    advantage: eqx.nn.Linear

    def __init__(self, input_size: int, output_size: int, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.linear1 = eqx.nn.Linear(input_size, 32, key=key1)
        self.linear2 = eqx.nn.Linear(32, 32, key=key2)
        self.linear3 = eqx.nn.Linear(32, 32, key=key3)
        self.linear4 = eqx.nn.Linear(32, 32, key=key4)
        self.state_value = eqx.nn.Linear(32, 1, key=key5)
        self.advantage = eqx.nn.Linear(32, output_size, key=key6)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        
        state_value = self.state_value(x)
        advantage = self.advantage(x)

        q_values = state_value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))

        return q_values

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

@eqx.filter_jit
def select_action(dqn: eqx.Module, obs: jnp.ndarray, epsilon: float, key) -> jnp.ndarray:
    '''
    selects an action using epsilon-greedy policy

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (jnp.ndarray): current observation
    - epsilon (float): exploration rate
    - key (jax.random.PRNGKey): random key for jax

    returns:
    - action: selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    expolation_actions = jax.random.randint(key, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1])
    is_greedy = jax.random.bernoulli(key, p=1-epsilon)
    chosen_action = jnp.where(is_greedy, greedy_actions, expolation_actions)

    return chosen_action


@eqx.filter_jit
def compute_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    '''
    computes the loss for the DQN

    Args:
    - q_network (eqx.Module): the DQN model
    - target_network (eqx.Module): the target DQN model
    - gamma (float): discount factor
    - batch (dict[str, jnp.ndarray]): batch of samples from the replay buffer

    Returns:
    - loss (jnp.ndarray): computed loss
    '''

    obs = batch['obs']
    next_obs = batch['next_obs']
    actions = batch['acts']
    rewards = batch['rews']
    done = batch['done']

    q_vals = eqx.filter_vmap(q_network)(obs)
    q_curr = jnp.take_along_axis(q_vals, actions[:, None], axis=-1).squeeze(-1)
    q_next = jnp.max(eqx.filter_vmap(target_network)(next_obs), axis=-1)

    mask = jnp.where(done, 0.0, 1.0)
    target = jax.lax.stop_gradient(rewards + gamma * q_next * mask)

    losses = optax.losses.huber_loss(q_curr, target)
    loss = jnp.mean(losses)

    return loss

def train_dqn(env: gym.Env, replay_buffer: ReplayBuffer, epsilon_scheduler, target_update_freq,
              gamma: float, q_network: eqx.Module, target_network: eqx.Module, optimizer: optax.GradientTransformation,
              optimizer_state, num_steps: int, key) -> tuple[eqx.Module, list[float], list[float], list[float]]:
    '''
    trains a DQN agent

    Args:
    - env (gym.Env): gym environment
    - replay_buffer (ReplayBuffer): replay buffer
    - epsilon_scheduler: epsilon decay schedule
    - target_update_freq (int): frequency of target network updates
    - gamma (float): discount factor
    - q_network (eqx.Module): Q-network
    - target_network (eqx.Module): target Q-network
    - optimizer (optax.GradientTransformation): optimizer
    - optimizer_state: state of the optimizer
    - num_steps (int): number of training steps
    - key: JAX random key

    Returns:
    - q_network (eqx.Module): trained Q-network
    - scores (list[float]): list of episode scores
    - losses (list[float]): list of losses
    - epsilons (list[float]): list of epsilon values
    '''
    scores = []
    losses = []
    epsilons = []
    episode_score = 0.0

    obs, _ = env.reset(seed=0)

    loss_and_grad = eqx.filter_value_and_grad(compute_loss)

    for step in range(num_steps):
        epsilon = epsilon_scheduler(step)
        epsilons.append(epsilon)
        key, subkey = jax.random.split(key)
        action = int(select_action(q_network, obs, epsilon, subkey).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_score += reward

        replay_buffer.store(obs, action, reward, next_obs, done)

        obs = next_obs

        if done:
            print(f"Step: {step}, Episode Score: {episode_score}, Epsilon: {epsilon:.3f}")
            scores.append(episode_score)
            obs, _ = env.reset()
            episode_score = 0.0

        if replay_buffer.can_sample():
            batch = replay_buffer.sample_batch()
            loss, grads = loss_and_grad(q_network, target_network, gamma, batch)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, q_network)
            q_network = eqx.apply_updates(q_network, updates)
            losses.append(loss.item())

        if step % target_update_freq == 0:
            target_network = eqx.tree_at(lambda m: m, target_network, q_network)

    return q_network, scores, losses, epsilons

def train_rainbow():
    raise NotImplementedError

def train_ppo():
    raise NotImplementedError

def train_a3c():
    raise NotImplementedError

if __name__ == "__main__":

    num_steps = 25000
    gamma = 0.99
    seed = 0
    target_update_freq = 500

    capacity = 10000
    batch_size = 128

    initial_epsilon = 1.0
    transition_steps = 10000
    final_epsilon = 0.1
    learning_rate = 1e-4
    max_norm = 1.0

    key = jax.random.key(seed)
    key, subkey1, subkey2, subkey4 = jax.random.split(key, 4)
    q_network = poll_agent(4, 2, subkey1)
    target_network = poll_agent(4, 2, subkey2)
    env = gym.make("CartPole-v1", max_episode_steps=1000)

    replay_buffer = ReplayBuffer(capacity, batch_size)

    epsilon_shed = optax.schedules.linear_schedule(initial_epsilon, final_epsilon,
        transition_steps, batch_size)

    optimizer = optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
    optimizer_state = optimizer.init(q_network)

    train_dqn(env, replay_buffer, epsilon_shed, target_update_freq, gamma,
                q_network, target_network, optimizer, optimizer_state, num_steps, key)

    env.close()
