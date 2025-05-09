import os
import equinox as eqx
import optax
import gymnasium as gym
import jax
from jax import value_and_grad, jit
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


def update_network(network: eqx.Module, optimizer: optax.GradientTransformation, optimizer_state: optax.OptState,
    loss_fn: callable, *loss_args) -> tuple[eqx.Module, float, optax.OptState]:
    loss, grads = value_and_grad(loss_fn)(network, *loss_args)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    network = optax.apply_updates(network, updates)
    
    return network, loss, optimizer_state


@dataclass(frozen=True)
class GroebnerState:
    ideal: Array
    selectables: Array


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


def plot_learning_process(scores: list[float], losses: list[float], epsilons: list[float]) -> None:
    '''
    Plots the training scores, losses, and epsilon values.

    Args:
    scores (List[float]): The training scores.
    losses (List[float]): The training losses.
    epsilons (List[float]): The epsilon values.

    Returns:
    None
    '''
    # smooth the losses
    smoothing_length_losses = 100
    smoothed_losses = scipy.signal.convolve(losses, np.ones(smoothing_length_losses) / smoothing_length_losses, mode='valid')
    smoothing_length_scores = 5
    smoothed_scores = scipy.signal.convolve(scores, np.ones(smoothing_length_scores) / smoothing_length_scores, mode='valid')

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f'score: {smoothed_scores[-1] if scores else 0:.2f}')
    plt.plot(smoothed_scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(smoothed_losses)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(133)
    plt.title('epsilons')
    plt.plot(epsilons)
    plt.xlabel("Update step")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




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


class poll_policy(eqx.Module):
    poll_value: poll_agent

    def __init__(self, input_size: int, output_size: int, key):
        self.poll_value = poll_agent(input_size, output_size, key)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        vals = self.poll_value(x)
        probs = jax.nn.softmax(vals)

        return probs
