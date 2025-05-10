import numpy as np
import matplotlib.pyplot as plt
from sympy.polys.domains import ZZ
from envs.ideals import random_ideal
from envs.deepgroebner import buchberger

def display_obs(obs):
    ideal, _ = obs
    print("\nIdeal:")
    for poly in ideal:
        print(f"{poly}")

def benchmark_game(strategy, step_limit, ideal):
    _, info = buchberger(ideal, selection=strategy, step_limit=step_limit)
    steps = info['steps']

    return steps

def benchmark_agent(strategy, num_episodes, step_limit, *ideal_params):
    step_counts = []
    for _ in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        step_count = benchmark_game(strategy, step_limit, ideal)
        step_counts.append(step_count)

    plot_pdf(step_counts, strategy)

def compare_agents(strategy1, strategy2, num_episodes, *ideal_params):
    step_diffs = []
    for _ in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        steps1 = benchmark_game(strategy1, ideal)
        steps2 = benchmark_game(strategy2, ideal)
        step_diffs.append(steps1 - steps2)
    
    plot_pdf(step_diffs, f'{strategy1}_vs_{strategy2}')

def plot_pdf(step_counts, strategy):
    var = np.var(step_counts)
    mean = np.mean(step_counts)
    plt.hist(step_counts, bins=100, density=True)
    plt.title(f'Variance: {var:.2f}, Mean: {mean:.2f}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Probability Density')
    plt.grid()
    plt.savefig(f'{strategy}.png')
    plt.close()


if __name__ == '__main__':
    num_episodes = 1000
    step_limit = 500
    ideal_params = [5, 7, 15, 3, ZZ, 'grevlex']

    benchmark_agent('degree_after_reduce', num_episodes, step_limit, *ideal_params)
