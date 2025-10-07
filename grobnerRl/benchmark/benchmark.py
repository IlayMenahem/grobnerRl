import os
import numpy as np
import matplotlib.pyplot as plt
from grobnerRl.envs.ideals import random_ideal
from grobnerRl.envs.deepgroebner import buchberger
from grobnerRl.benchmark.optimalReductions import optimal_reductions
from typing import Optional


def benchmark_assistanted_game(strategy: str, ideal, step_limit: Optional[int]) -> tuple[list, Optional[int]]:
    '''
    Benchmarks a given strategy for Buchberger's algorithm with the assistance of the
    ideal generators being added one of the groebner basis generators.

    Args:
    - strategy (str): The selection strategy to use.
    - num_episodes (int): The number of episodes to simulate.
    - step_limit (Optional[int]): The maximum number of steps per game.
    - *ideal_params (list): Parameters to be passed to `random_ideal` for generating ideals.

    Returns: tuple[list, int]: A tuple containing:
    - basis (list): The computed Gröbner basis.
    - steps (int): The number of steps taken to compute the basis.
    '''
    basis, _ = buchberger(ideal, selection=strategy, step_limit=step_limit)

    ideal = ideal + basis[-len(basis)//2:]

    basis, info = buchberger(ideal, selection=strategy, step_limit=step_limit)

    steps: int = info['total_reward']

    return basis, steps


def benchmark_agent(agent, num_episodes: int, env, folder='figs', agent_name: Optional[str] = None) -> None:
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


def compare_agents(strategy1, strategy2, num_episodes: int, *ideal_params: list) -> None:
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

    plot_pdf(step_diffs, f'{strategy1}_vs_{strategy2}')


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
    plt.title(f'Variance: {var:.2f}, Mean: {mean:.2f}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Probability Density')
    plt.grid()
    plt.savefig(f'{strategy_name}.png')
    plt.close()


def optimal_vs_standard(num_episodes: int, *ideal_params: list) -> None:
    '''
    compare the optimal reductions with the standard Buchberger's algorithm
    '''
    points: list[tuple[int,int]] = []
    optimal_fail: int = 0

    for _ in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        _, reds, _ = optimal_reductions(ideal, 10000)
        _, normal_reds = buchberger(ideal)

        if not reds:
            optimal_fail += 1
            continue

        if len(reds) > len(normal_reds):
            print('problematic ideal')
            display_obs((ideal, []))

        points.append((len(reds), len(normal_reds)))

    print(f'optimal reductions failed {optimal_fail} times')

    plt.scatter(*zip(*points), alpha=0.5)
    plt.xlabel('optimal reductions')
    plt.ylabel('standard reductions')
    plt.savefig(os.path.join('figs', 'optimal_vs_standard.png'))
    plt.close()
