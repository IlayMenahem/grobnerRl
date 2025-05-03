import jax
import optax
from tqdm import tqdm
import equinox as eqx
import gymnasium as gym

from .selector import select_action
from .utils import plot_learning_process

def learner_step(replay_buffer, gamma, q_network, target_network, optimizer, optimizer_state, loss_fn):
    loss_and_grad = eqx.filter_value_and_grad(loss_fn)

    batch = replay_buffer.sample_batch()
    loss, grads = loss_and_grad(q_network, target_network, gamma, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, q_network)
    q_network = eqx.apply_updates(q_network, updates)

    return q_network, loss, optimizer_state, optimizer


def train_dqn(env, replay_buffer, epsilon_scheduler, target_update_freq, gamma: float,
    q_network: eqx.Module, target_network: eqx.Module, optimizer: optax.GradientTransformation,
    optimizer_state, num_steps: int, loss_fn, key) -> tuple[eqx.Module, list[float], list[float], list[float]]:
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

    Returns:
    - q_network (eqx.Module): trained Q-network
    - scores (list[float]): list of episode scores
    - losses (list[float]): list of losses
    - epsilons (list[float]): list of epsilon values
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
        action = select_action(q_network, obs, epsilon, subkey)

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
            q_network, loss, optimizer_state, optimizer = learner_step(replay_buffer, gamma,
                q_network, target_network, optimizer, optimizer_state, loss_fn)
            losses.append(loss.item())

        if step % target_update_freq == 0:
            target_network = eqx.tree_at(lambda m: m, target_network, q_network)

    progress_bar.close()
    plot_learning_process(scores, losses, epsilons)

    return q_network, scores, losses, epsilons


def train_ppo(env: gym.Env, policy: eqx.Module, critic: eqx.Module):
    raise NotImplementedError

def train_a3c(env: gym.Env, policy: eqx.Module, critic: eqx.Module, gamma: float, n_steps: int):
    raise NotImplementedError
