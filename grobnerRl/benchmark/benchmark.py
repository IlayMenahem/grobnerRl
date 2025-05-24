import os
import numpy as np
import matplotlib.pyplot as plt
from grobnerRl.envs.ideals import random_ideal
from grobnerRl.envs.deepgroebner import buchberger
from grobnerRl.benchmark.optimalReductions import optimal_reductions
from typing import Optional


def benchmark_game(strategy: str, ideal: list, step_limit: Optional[int] = None) -> tuple[list, int]:
    '''
    Simulates a single game of Buchberger's algorithm using the specified strategy.

    Args:
        strategy (str): The selection strategy to use for Buchberger's algorithm.
        ideal (list[Any]): The initial ideal represented as a list of polynomials.
        step_limit (Optional[int]): The maximum number of steps to allow in the algorithm.
                                    Can be None for no limit; None by default.

    Returns:
        tuple[list, int]: A tuple containing:
            - basis (list): The computed Gröbner basis.
            - steps (int): The number of steps taken to compute the basis.
    '''
    basis, info = buchberger(ideal, selection=strategy, step_limit=step_limit)

    if not info['valid']:
        return [], None

    steps: int = info['steps']

    return basis, steps


def benchmark_agent(strategy: str, num_episodes: int, step_limit: Optional[int], *ideal_params: list) -> None:
    """
    Benchmarks a given strategy for Buchberger's algorithm over multiple episodes.

    For each episode, a random ideal is generated, and the game is simulated
    using the specified strategy. The distribution of step counts is then plotted.

    Args:
        strategy (str): The selection strategy to use.
        num_episodes (int): The number of episodes to simulate.
        step_limit (Optional[int]): The maximum number of steps per game.
        *ideal_params (list): Parameters to be passed to `random_ideal` for
                             generating ideals.
    """
    step_counts: list[int] = []
    for _ in range(num_episodes):
        current_ideal: list = random_ideal(*ideal_params)
        _, step_count = benchmark_game(strategy, current_ideal, step_limit)
        step_counts.append(step_count)

    plot_pdf(step_counts, strategy)


def compare_agents(strategy1: str, strategy2: str, num_episodes: int, *ideal_params: list) -> None:
    """
    Compares two strategies for Buchberger's algorithm over multiple episodes.

    For each episode, a random ideal is generated, and the game is simulated
    using both strategies on the same ideal. The distribution of the differences
    in step counts (steps1 - steps2) is then plotted.

    Args:
        strategy1 (str): The first selection strategy.
        strategy2 (str): The second selection strategy.
        num_episodes (int): The number of episodes to simulate for comparison.
        *ideal_params (Any): Parameters to be passed to `random_ideal` for
                             generating ideals.
    """
    step_diffs: list[int] = []
    for _ in range(num_episodes):
        current_ideal: list = random_ideal(*ideal_params)
        _, steps1 = benchmark_game(strategy1, current_ideal)
        _, steps2 = benchmark_game(strategy2, current_ideal)
        step_diffs.append(steps1 - steps2)

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


def plot_pdf(step_counts: list[int], strategy_name: str) -> None:
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
    points = []
    optimal_fail = 0
    standard_fail = 0

    for _ in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        _, reds, _ = optimal_reductions(ideal, 10000)
        _, num_steps = benchmark_game('normal', ideal, 1000)

        if not reds:
            optimal_fail += 1

        if not num_steps:
            standard_fail += 1

        if not reds or not num_steps:
            continue

        if len(reds) > num_steps:
            print('problematic ideal')
            display_obs((ideal, []))

        points.append((len(reds), num_steps))

    print(f'optimal reductions failed {optimal_fail} times')
    print(f'standard reductions failed {standard_fail} times')
    print(len(points))

    sns.set_theme(style="whitegrid")
    plt.scatter(*zip(*points), alpha=0.5)
    plt.xlabel('optimal reductions')
    plt.ylabel('standard reductions')
    plt.savefig(os.path.join('figs', 'optimal_vs_standard.png'))
