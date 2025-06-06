import os
import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
import numpy as np
from dataclasses import dataclass
from chex import Array


def select_action_inference(dqn: eqx.Module, obs: Array) -> tuple[int, ...]:
    '''
    selects an action using the DQN model

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (Array): current observation

    returns:
    - action (tuple[int, ...]): selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    chosen_action = jnp.array(jnp.unravel_index(jnp.argmax(q_vals), q_vals.shape))

    chosen_action = tuple(i.item() for i in chosen_action)

    return chosen_action


def select_action_policy(policy: eqx.Module, obs: Array, key: Array) -> int|tuple[int, ...]:
    probs = policy(obs)
    probs_flat = jnp.reshape(probs, -1)
    action = jax.random.choice(key, probs_flat.shape[0], p=probs_flat)

    if probs.ndim > 1:
        action = jnp.unravel_index(action, probs.shape)
        return tuple(i.item() for i in action)
    else:
        return action.item()


def update_network(network: eqx.Module, optimizer: optax.GradientTransformation, optimizer_state: optax.OptState,
    loss_fn: callable, *loss_args) -> tuple[eqx.Module, float, optax.OptState]:
    loss, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True)(network, *loss_args)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    network = eqx.apply_updates(network, updates)

    return network, loss, optimizer_state


@dataclass(frozen=True)
class GroebnerState:
    ideal: list[np.ndarray]
    selectables: list[tuple]


@dataclass(frozen=True)
class TimeStep:
    obs: GroebnerState
    action: tuple[int, ...] | int
    reward: float
    next_obs: GroebnerState
    done: bool


def callback_save_model(model, directory: str, filename: str) -> None:
    '''
    saves the model to the specified directory

    Args:
    - model (eqx.Module): the model to save
    - directory (str): the directory to save the model to
    - filename (str): the name of the file to save the model to
    '''
    path = os.path.join(directory, filename)
    eqx.tree_serialise_leaves(path, model)


def callback_eval(model, env, num_episodes: int) -> float:
    '''
    evaluates the model on the specified environment

    Args:
    - model (eqx.Module): the model to evaluate
    - env (gym.Env): the environment to evaluate the model on
    - num_episodes (int): number of episodes to evaluate the model on

    Returns:
    - average_reward (float): average reward over the evaluated episodes
    '''
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action = select_action_inference(model, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    mean_reward = total_reward / num_episodes

    return mean_reward


def plot_learning_process(scores: list[float], vals1: list[float], vals2: list[float]) -> None:
    '''
    Plots the training scores, losses, and epsilon values.

    Args:
    scores (List[float]): The training scores.
    losses (List[float]): The training losses.
    epsilons (List[float]): The epsilon values.

    Returns:
    None
    '''

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    window = 50
    plt.plot(scores)
    if len(scores) >= window:
        running_avg = [sum(scores[max(0, i - window):i]) / window for i in range(len(scores))]
        plt.plot(running_avg, label=f'Running Avg (window={window})')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(vals1)
    plt.xlabel("Update step")
    plt.grid(True)
    plt.subplot(133)
    plt.plot(vals2)
    plt.xlabel("Update step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
