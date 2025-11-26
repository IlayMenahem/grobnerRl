import copy
import itertools
import multiprocessing
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from grobnerRl.envs.env import BaseEnv
from grobnerRl.experts import Expert


def benchmark_agent(
    agent, num_episodes: int, env, folder="figs", agent_name: Optional[str] = None
) -> None:
    """
    Benchmarks a given agent for Buchberger's algorithm over multiple episodes.

    For each episode, a random ideal is generated, and the game is simulated
    using the agent's .act() method. The distribution of step counts is then plotted.

    Args:
    agent: An agent object with an .act() method that takes an observation and returns an action.
    num_episodes (int): The number of episodes to simulate.
    step_limit (Optional[int]): The maximum number of steps per game.
    env: The environment to use for the simulation.
    folder (str): The folder where the plot will be saved. Default is 'figs'.
    agent_name (Optional[str]): Name for the agent, used in plot filename. If None, uses agent class name.
    """
    reward_counts: list[int] = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        reward_total = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += reward

        reward_counts.append(reward_total)

    name = agent_name if agent_name is not None else agent.__class__.__name__
    file_name = os.path.join(folder, name)
    plot_pdf(reward_counts, file_name)


def _simulate_episode(env: BaseEnv, expert: Expert) -> int:
    obs, _ = env.reset()
    expert.env = env

    _, pairs = obs
    if not pairs:
        return 0

    reward_total = 0
    terminated, truncated = False, False

    while not (terminated or truncated):
        action = expert(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_total += reward

    return reward_total


def benchmark_expert(
    expert: Expert, num_episodes: int, env: BaseEnv, folder="figs"
) -> None:
    """
    Benchmarks a given expert for Buchberger's algorithm over multiple episodes.

    Args:
    - expert (Expert): An expert object.
    - num_episodes (int): The number of episodes to simulate.
    - env (BaseEnv): The environment to use for the simulation.
    - folder (str): The folder where the plot will be saved. Default is 'figs'.
    """
    with multiprocessing.Pool() as pool:
        # deepcopy the env to ensure each process has a fresh copy
        envs = (copy.deepcopy(env) for _ in range(num_episodes))
        experts = itertools.repeat(expert, num_episodes)
        reward_counts = pool.starmap(
            _simulate_episode,
            zip(envs, experts),
        )

    file_name = os.path.join(folder, expert.__class__.__name__)
    plot_pdf(reward_counts, file_name)


def compare_agents(
    strategy1, strategy2, num_episodes: int, *ideal_params: list
) -> None:
    """
    Compares two strategies for Buchberger's algorithm over multiple episodes.

    For each episode, a random ideal is generated, and the game is simulated
    using both strategies on the same ideal. The distribution of the differences
    in step counts (steps1 - steps2) is then plotted.

    Args:
    agent1: The first selection strategy.
    agent2: The second selection strategy.
    num_episodes (int): The number of episodes to simulate for comparison.
    *ideal_params (Any): Parameters to be passed to `random_ideal` for
                            generating ideals.
    """
    step_diffs: list[int] = []

    for _ in range(num_episodes):
        raise NotImplementedError("This function needs to be implemented.")

    plot_pdf(step_diffs, f"{strategy1}_vs_{strategy2}")


def display_obs(obs: tuple[list, list]) -> None:
    """
    Prints the components of an observation (ideal and selectables).

    Args:
        obs (tuple[list, list]): A tuple containing:
            - ideal (list): The current list of polynomials in the ideal.
            - selectables (list): The list of selectable S-polynomial pairs.
    """
    ideal, selectables = obs

    print("\nIdeal:")
    for poly in ideal:
        print(poly.as_expr())

    print("\nSelectables:")
    for selectable in selectables:
        print(selectable)


def plot_pdf(step_counts: list, strategy_name: str) -> None:
    """
    Plots and saves a probability density function (histogram) of step counts.

    The plot shows the distribution of step counts, along with the mean and
    variance. The plot is saved to a PNG file named after the strategy.

    Args:
        step_counts (List[int]): A list of step counts from simulations.
        strategy_name (str): The name of the strategy or comparison, used for
                             the plot title and filename.
    """
    var: float = np.var(step_counts)
    mean: float = np.mean(step_counts)
    plt.hist(step_counts, bins=100, density=True)
    plt.title(f"Variance: {var:.2f}, Mean: {mean:.2f}")
    plt.xlabel("Number of Steps")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.savefig(f"{strategy_name}.png")
    plt.close()
