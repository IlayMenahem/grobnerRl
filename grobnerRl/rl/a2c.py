from collections import deque
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import equinox as eqx

from grobnerRl.rl.utils import TimeStep, GroebnerState, update_network, plot_learning_process, select_action_policy


class TransitionSet:
    queue: deque[TimeStep]
    size: int

    def __init__(self, size: int) -> None:
        self.queue = deque(maxlen=size)
        self.size = size

    def store(self, obs: GroebnerState, act: tuple[int, ...] | int, rew: float, next_obs: GroebnerState, done: bool) -> None:
        self.queue.append(TimeStep(obs, act, rew, next_obs, done))

    def sample_and_clear(self) -> tuple[list,...]:
        res = ([t.obs for t in self.queue],
            [t.action for t in self.queue],
            [jnp.array(t.reward) for t in self.queue],
            [t.next_obs for t in self.queue],
            [jnp.array(t.done) for t in self.queue])
        self.queue = deque(maxlen=self.size)

        return res


def value_loss(critic: eqx.Module, gamma: float, batch: tuple):
    state, _, reward, next_state, done = batch

    def value_loss_fn(reward, state, next_state, done):
        value = critic(state)
        next_value = critic(next_state)

        target = reward + gamma * next_value * (1 - done)

        loss = optax.huber_loss(value, target, delta=1.0)

        return loss

    loss = jax.tree.map(value_loss_fn, reward, state, next_state, done, is_leaf=lambda x: not isinstance(x, list))
    loss = jnp.mean(jnp.array(loss))

    return loss


def policy_loss(policy, critic, gamma, batch, entropy_coeff):
    states, actions, rewards, next_states, done = batch

    def policy_loss_fn(reward, action, state, next_state, done):
        value = critic(state)
        next_value = critic(next_state)

        target = reward + gamma * next_value * (1 - done)
        advantage = jax.lax.stop_gradient(target - value)

        action_probs = policy(state)
        log_prob = jnp.log(action_probs[action])

        policy_loss_val = -log_prob * advantage
        entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8))

        loss = policy_loss_val - entropy_coeff * entropy

        return loss

    loss = jax.tree.map(policy_loss_fn, rewards, actions, states, next_states, done, is_leaf=lambda x: not isinstance(x, list))
    loss = jnp.mean(jnp.array(loss))

    return loss


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
            print(f"Episode score: {episode_score}")
            obs, _ = env.reset()
            scores.append(episode_score)
            episode_score = 0.0

    return env, replay_buffer, policy, key, scores, episode_score, obs


def train_a2c(env, policy: eqx.Module, critic: eqx.Module, optimizer_policy, optimizer_critic,
    gamma: float, num_episodes: int, n_steps: int, key, entropy_coeff: float = 0.01) -> tuple[eqx.Module, eqx.Module, list[float], list[tuple[float, float]]]:
    '''
    Train an Advantage Actor-Critic (A2C) agent.

    This function implements the A2C algorithm, which combines policy gradient methods
    with value function approximation. The actor (policy) learns to select actions
    while the critic (value function) estimates state values to reduce variance.

    Args:
        env: The environment to train on
        policy: The actor network (policy) as an Equinox module
        critic: The critic network (value function) as an Equinox module
        optimizer_policy: Optax optimizer for the policy network
        optimizer_critic: Optax optimizer for the critic network
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        num_episodes: Number of training episodes to run
        n_steps: Number of environment steps to collect per episode before updating
        key: JAX random key for stochastic operations
        entropy_coeff: Coefficient for entropy regularization (default: 0.01)

    Returns:
        tuple containing:
            - policy: Updated policy network
            - critic: Updated critic network
            - scores: List of episode scores achieved during training
            - losses: List of tuples containing (actor_loss, critic_loss) for each episode
    '''
    replay_buffer = TransitionSet(n_steps)
    optimizer_critic_state = optimizer_critic.init(critic)
    optimizer_policy_state = optimizer_policy.init(policy)

    scores = []
    losses = []

    progress_bar = tqdm(total=num_episodes, unit="episode")
    episode_score = 0.0
    obs, _ = env.reset()

    for episode in range(num_episodes):
        env, replay_buffer, policy, key, scores, episode_score, obs = collect_transitions(env, replay_buffer, policy, n_steps, key, scores, episode_score, obs)

        batch = replay_buffer.sample_and_clear()

        policy, actor_loss, optimizer_policy_state = update_network(policy, optimizer_policy, optimizer_policy_state, policy_loss, critic, gamma, batch, entropy_coeff)
        critic, critic_loss, optimizer_critic_state = update_network(critic, optimizer_critic, optimizer_critic_state, value_loss, gamma, batch)

        losses.append((actor_loss, critic_loss))
        progress_bar.set_postfix(loss=actor_loss, critic_loss=critic_loss)
        progress_bar.update(1)

    progress_bar.close()
    policy_losses, critic_losses = map(list, zip(*losses))
    plot_learning_process(scores, policy_losses, critic_losses)

    return policy, critic, scores, losses
