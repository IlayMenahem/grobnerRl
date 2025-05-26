from collections import deque
import jax
from jax import vmap, jit
import jax.numpy as jnp
import optax
from tqdm import tqdm
import equinox as eqx
import gymnasium as gym
from chex import Array

from grobnerRl.rl.utils import TimeStep, GroebnerState, update_network


class TransitionSet:
    queue: deque[TimeStep]
    size: int

    def __init__(self, size: int) -> None:
        self.queue = deque(maxlen=size)
        self.size = size

    def store(self, obs: GroebnerState, act: tuple[int, ...] | int, rew: float, next_obs: GroebnerState, done: bool) -> None:
        self.queue.append(TimeStep(obs, act, rew, next_obs, done))

    def sample_and_clear(self) -> tuple[Array, Array, Array, Array, Array]:
        res = (jnp.array([t.obs for t in self.queue]),
            jnp.array([t.action for t in self.queue]),
            jnp.array([t.reward for t in self.queue]),
            jnp.array([t.next_obs for t in self.queue]),
            jnp.array([t.done for t in self.queue]))
        self.queue = deque(maxlen=self.size)

        return res


@jit
def compute_value_and_target(critic, reward, gamma, state, next_state, done):
    value = critic(state)
    next_value = critic(next_state)

    target = reward + gamma * next_value * (1 - done)

    return value, target


@jit
def advantage_loss(critic: eqx.Module, gamma: float, batch: tuple):
    state, _, reward, next_state, done = batch

    value, target = vmap(compute_value_and_target, in_axes=(None, 0, None, 0, 0, 0))(critic, reward, gamma, state, next_state, done)
    loss = jnp.mean(optax.l2_loss(value, target))

    return loss


@jit
def policy_loss(policy, critic, gamma, batch):
    states, actions, rewards, next_states, done = batch

    def policy_loss_fn(policy, critic, state, next_state, action, gamma, reward, done):
        value, target = compute_value_and_target(critic, reward, gamma, state, next_state, done)
        advantage = target - value
        log_prob = jnp.log(policy(state)[action])

        return -log_prob * advantage

    loss = vmap(policy_loss_fn, in_axes=(None, None, 0, 0, 0, None, 0, 0))(policy, critic, states, next_states, actions, gamma, rewards, done)
    loss = jnp.mean(loss)

    return loss


def select_action_policy(policy: eqx.Module, obs: Array, key: Array) -> int:
    probs = policy(obs)
    n = probs.shape[-1]
    action = jax.random.choice(key, n, p=probs)
    action = action.item()

    return action


def collect_transitions(env, replay_buffer, policy, n_steps, key, scores, episode_score, obs):
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        action = select_action_policy(policy, obs, subkey)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_score += reward

        replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()
            scores.append(episode_score)
            episode_score = 0.0

    return env, replay_buffer, policy, key, scores, episode_score, obs


def train_a2c(env: gym.Env, replay_buffer: TransitionSet, policy: eqx.Module, critic: eqx.Module,
    optimizer_policy, optimizer_policy_state, optimizer_critic, optimizer_critic_state,
    gamma: float, num_episodes: int, n_steps: int, key) -> tuple[eqx.Module, eqx.Module, list[float], list[tuple[float, float]]]:
    '''
    Train an Advantage Actor-Critic (A2C) agent.

    This function implements the A2C algorithm, which combines policy gradient methods
    with value function approximation. The actor (policy) learns to select actions
    while the critic (value function) estimates state values to reduce variance.

    Args:
        env: The environment to train on
        replay_buffer: TransitionSet buffer to store and sample experience transitions
        policy: The actor network (policy) as an Equinox module
        critic: The critic network (value function) as an Equinox module
        optimizer_policy: Optax optimizer for the policy network
        optimizer_policy_state: State of the policy optimizer
        optimizer_critic: Optax optimizer for the critic network
        optimizer_critic_state: State of the critic optimizer
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        num_episodes: Number of training episodes to run
        n_steps: Number of environment steps to collect per episode before updating
        key: JAX random key for stochastic operations

    Returns:
        tuple containing:
            - policy: Updated policy network
            - critic: Updated critic network
            - scores: List of episode scores achieved during training
            - losses: List of tuples containing (actor_loss, critic_loss) for each episode
    '''
    scores = []
    losses = []

    progress_bar = tqdm(total=num_episodes, unit="episode")
    episode_score = 0.0
    obs, _ = env.reset()

    for episode in range(num_episodes):
        env, replay_buffer, policy, key, scores, episode_score, obs = collect_transitions(env, replay_buffer, policy, n_steps, key, scores, episode_score, obs)

        batch = replay_buffer.sample_and_clear()
        policy, actor_loss, optimizer_policy_state = update_network(policy, optimizer_policy, optimizer_policy_state, policy_loss, critic, gamma, batch)
        critic, critic_loss, optimizer_critic_state = update_network(critic, optimizer_critic, optimizer_critic_state, advantage_loss, gamma, batch)

        losses.append((actor_loss, critic_loss))
        progress_bar.set_postfix(loss=actor_loss, critic_loss=critic_loss)
        progress_bar.update(1)

    progress_bar.close()

    return policy, critic, scores, losses
