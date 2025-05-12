from collections import deque
from functools import partial
import random

import jax
from jax import vmap
import jax.numpy as jnp
import optax
from tqdm import tqdm
import equinox as eqx
from chex import Array

from grobnerRl.rl.utils import TimeStep, GroebnerState, plot_learning_process



def td_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, obs: GroebnerState,
    next_obs: GroebnerState, action, reward, done) -> Array:
    q_vals = q_network(obs)
    q_curr = q_vals[action]

    target_q_next = target_network(next_obs)
    q_next = jnp.max(target_q_next)

    mask = jnp.where(done, 0.0, 1.0)
    target = jax.lax.stop_gradient(reward + gamma * q_next * mask)

    loss = optax.losses.huber_loss(q_curr, target)

    return loss


@jax.jit
def dqn_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, batch: dict) -> jnp.ndarray:
    '''
    computes the loss for the Double DQN

    Args:
    - q_network (eqx.Module): the DQN model
    - target_network (eqx.Module): the target DQN model
    - gamma (float): discount factor
    - batch (dict): batch of samples from the replay buffer

    Returns:
    - loss (jnp.ndarray): computed loss
    '''

    observations: list[GroebnerState] = batch['obs']
    next_observations: list[GroebnerState] = batch['next_obs']
    actions: list[Array] = batch['acts']
    rewards: list[Array] = batch['rews']
    dones: list[Array] = batch['done']

    td_wrapper = partial(td_loss, q_network, target_network, gamma)
    
    losses = vmap(td_wrapper)(observations, next_observations, actions, rewards, dones)
    loss = jnp.mean(losses)

    return loss


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

        batch = {'obs': jnp.array([t.obs for t in samples]),
                'next_obs': jnp.array([t.next_obs for t in samples]),
                'acts': jnp.array([t.action for t in samples]),
                'rews': jnp.array([t.reward for t in samples]),
                'done': jnp.array([t.done for t in samples])}

        return batch

    def can_sample(self) -> bool:
        return len(self.queue) >= self.batch_size


def uniform_sample_index(key: Array, mask: Array):
    '''
    samples an array index uniformly from the array

    Args:
    - key (Array): random key for jax
    - mask (Array): mask of legal indices

    returns:
    - result (Array | int): sampled index
    '''
    flat_mask = mask.reshape(-1)
    flat_length = len(flat_mask)
    flat_indices = jnp.arange(flat_length)[flat_mask]
    sampled_index = jax.random.choice(key, flat_indices)
    sampled_index = jnp.unravel_index(sampled_index, mask.shape)
    sampled_index = jnp.array(sampled_index)

    return sampled_index


def select_action_epsilon(dqn: eqx.Module, obs: Array, epsilon: float, key: Array) -> tuple[int, ...]:
    '''
    selects an action using epsilon-greedy policy

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (Array): current observation, can be n dimensional
    - epsilon (float): exploration rate
    - key (Array): random key for jax

    returns:
    - action (tuple[int, ...]): selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    is_legal = jnp.isfinite(q_vals)

    greedy_action = jnp.array(jnp.unravel_index(jnp.argmax(q_vals), q_vals.shape))
    epsilon_action = uniform_sample_index(key, is_legal)
    chosen_action = jax.lax.select(jax.random.uniform(key) < epsilon, epsilon_action, greedy_action)

    # chosen_action = tuple(i.item() for i in chosen_action)
    chosen_action = chosen_action.item()

    return chosen_action


def learner_step(batch, gamma, q_network, target_network, optimizer, optimizer_state, 
    loss_fn) -> tuple[eqx.Module, jnp.ndarray, optax.OptState, optax.GradientTransformation]:
    loss, grads = jax.value_and_grad(loss_fn)(q_network, target_network, gamma, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, q_network)
    q_network = optax.apply_updates(q_network, updates)

    return q_network, loss, optimizer_state, optimizer


def train_dqn(env, replay_buffer: ReplayBuffer, epsilon_scheduler: callable, 
    target_update_freq: int, gamma: float, q_network: eqx.Module, 
    target_network: eqx.Module, optimizer: optax.GradientTransformation,
    optimizer_state, num_steps: int, loss_fn, key) -> eqx.Module:
    '''
    trains a DQN agent

    Args:
    - env: an environment
    - replay_buffer: replay buffer
    - epsilon_scheduler: epsilon decay schedule
    - target_update_freq (int): frequency of target network updates
    - gamma (float): discount factor
    - q_network (eqx.Module): Q-network
    - target_network (eqx.Module): target Q-network
    - optimizer (optax.GradientTransformation): optimizer
    - optimizer_state: state of the optimizer
    - num_steps (int): number of training steps
    - loss_fn: loss function
    - key: JAX random key

    Returns: q_network (eqx.Module): trained Q-network
    '''
    scores = []
    losses = []
    epsilons = []
    progress_bar = tqdm(total=num_steps, unit="step")
    episode_score = 0.0

    obs, _ = env.reset()

    for step in range(num_steps):
        epsilon = epsilon_scheduler(step)
        epsilons.append(epsilon)
        key, subkey = jax.random.split(key)
        action = select_action_epsilon(q_network, obs, epsilon, subkey)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_score += reward

        replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs
        progress_bar.update(1)

        if done:
            scores.append(episode_score)
            episode_score = 0.0
            obs, _ = env.reset()

        if replay_buffer.can_sample():
            batch = replay_buffer.sample_batch()
            q_network, loss, optimizer_state, optimizer = learner_step(batch, gamma,
                q_network, target_network, optimizer, optimizer_state, loss_fn)
            losses.append(loss.item())

        if step % target_update_freq == 0:
            target_network = eqx.tree_at(lambda m: m, target_network, q_network)

    progress_bar.close()
    plot_learning_process(scores, losses, epsilons)

    return q_network
