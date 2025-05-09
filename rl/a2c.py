from collections import deque
import jax
from jax import vmap, jit
import jax.numpy as jnp
import optax
from tqdm import tqdm
import equinox as eqx
from chex import Array
import gymnasium as gym

from .utils import TimeStep, GroebnerState, update_network, poll_agent, poll_policy


class TransitionSet:
    queue: deque[TimeStep]
    size: int

    def __init__(self, size: int) -> None:
        self.queue = deque(maxlen=size)
        self.size = size

    def store(self, obs: GroebnerState, act: tuple[int, ...] | int, rew: float, next_obs: GroebnerState, done: bool) -> None:
        self.queue.append(TimeStep(obs, act, rew, next_obs, done))

    def sample_and_clear(self) -> deque[TimeStep]:
        res = (jnp.array([t.obs for t in self.queue]),
            jnp.array([t.action for t in self.queue]),
            jnp.array([t.reward for t in self.queue]),
            jnp.array([t.next_obs for t in self.queue]),
            jnp.array([t.done for t in self.queue]))
        self.queue = deque(maxlen=self.size)

        return res


@jit
def advantage(critic, reward, gamma, state, next_state, done):
    value = critic(state)
    next_value = critic(next_state)

    target = reward + gamma * next_value * (1 - done)

    return value, target


@jit
def advantage_loss(critic: eqx.Module, gamma: float, batch: tuple):
    state, _, reward, next_state, done = batch

    value, target = vmap(advantage, in_axes=(None, 0, None, 0, 0, 0))(critic, reward, gamma, state, next_state, done)
    loss = jnp.mean(optax.l2_loss(value, target))

    return loss


@jit
def policy_loss(policy, critic, gamma, batch):
    states, actions, rewards, next_states, done = batch

    def policy_loss_fn(policy, critic, state, next_state, action, gamma, reward, done):
        _, adva = advantage(critic, reward, gamma, state, next_state, done)
        log_prob = jnp.log(policy(state)[action])
        
        return -adva * log_prob
    
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


def train_a2c(env, replay_buffer: TransitionSet, policy: eqx.Module, critic: eqx.Module, 
    optimizer_policy, optimizer_policy_state, optimizer_critic, optimizer_critic_state, 
    gamma: float, num_episodes: int, n_steps: int, key) -> tuple[eqx.Module, eqx.Module, list[float], list[tuple[float, float]]]:
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


if __name__ == "__main__":
    num_episodes = 1000
    n_steps = 256
    gamma = 0.9
    seed = 0
    key = jax.random.key(seed)

    env = gym.make('Acrobot-v1', max_episode_steps=250)
    replay_buffer = TransitionSet(n_steps)
    policy = poll_policy(6, 3, key)
    critic = poll_agent(6, 1, key)

    optimizer_policy = optax.adam(1e-4)
    optimizer_policy_state = optimizer_policy.init(policy)
    optimizer_critic = optax.adam(1e-3)
    optimizer_critic_state = optimizer_critic.init(critic)

    policy, critic, scores, losses = train_a2c(env, replay_buffer, policy, critic, 
        optimizer_policy, optimizer_policy_state, optimizer_critic, optimizer_critic_state, 
        gamma, num_episodes, n_steps, key)
    
    env.close()
    print(scores)
