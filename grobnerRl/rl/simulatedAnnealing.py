import math
import random
import numpy as np
from functools import lru_cache
from scipy.special import softmax
from itertools import combinations, permutations
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_temp(T0, k, type='log'):
    if type == 'log':
        return T0/math.log(k + 2)
    elif type == 'linear':
        return T0/(k + 1)

    raise ValueError(f"Unknown temperature type: {type}. Supported types are 'log' and 'linear'.")


def _simulated_annealing(t0: float, num_steps: int, state0, neighbor_fn, cost_fn, next_state_fn, minimize=True):
    '''
    Perform simulated annealing to find a near-optimal solution.

    Parameters:
    - t0 (float): Initial temperature.
    - num_steps (int): Number of steps to perform.
    - state0: Initial state.
    - neighbor_fn (function): Function to generate a neighbor state.
    - cost_fn (function): Function to compute the cost of a state.
    - minimize (bool): If True, minimize the cost; if False, maximize the cost.
    '''
    current_state = state0
    current_cost = cost_fn(current_state)
    best_state = current_state
    best_cost = current_cost

    progress_best = [current_cost]
    progress_current = [current_cost]
    temperatures = []

    pbar = tqdm(range(num_steps))
    update_every = num_steps // 10000

    for k in range(num_steps):
        temperature = get_temp(t0, k)

        current_state, current_cost = next_state_fn(neighbor_fn, current_state, cost_fn, current_cost, minimize, temperature)

        if current_cost < best_cost:
            best_state = current_state
            best_cost = current_cost

        progress_best.append(best_cost)
        progress_current.append(current_cost)
        temperatures.append(temperature)

        if k % update_every == 0:
            pbar.update(update_every)
            pbar.set_description(f'temp {temperature:.2f}, best {best_cost:.2f}')

    return best_state, best_cost, progress_best, progress_current, temperatures


def next_state_sa(neighbor_fn, current_state, cost_fn, current_cost, minimize, temperature):
    neighbor_state = neighbor_fn(current_state)
    neighbor_cost = cost_fn(neighbor_state)

    delta_cost = neighbor_cost - current_cost
    if not minimize:
        neighbor_cost = -neighbor_cost

    if (delta_cost > 0 and minimize) or (delta_cost < 0 and not minimize):
        probability = math.exp(-delta_cost / temperature)
    else:
        probability = 1.0

    if random.random() < probability:
        current_state = neighbor_state
        current_cost = neighbor_cost

    return current_state, current_cost


def next_state_diffusion(neighbors_fn, current_state, cost_fn, current_cost, minimize, temperature):
    neighbors_state, neighbors_cost = neighbors_fn(current_state)

    weights = current_cost - neighbors_cost / temperature
    if not minimize:
        weights = -weights

    probabilities = softmax(weights)
    chosen_index = np.random.choice(len(neighbors_state), p=probabilities)

    current_state = neighbors_state[chosen_index]
    current_cost = neighbors_cost[chosen_index]

    return current_state, current_cost


def simulated_annealing(t0: float, num_steps: int, state0, neighbor_fn, cost_fn, minimize=True):
    return _simulated_annealing(t0, num_steps, state0, neighbor_fn, cost_fn, next_state_sa, minimize)


def simulated_diffusion(t0: float, num_steps: int, state0, neighbors_fn, cost_fn, minimize=True):
    return _simulated_annealing(t0, num_steps, state0, neighbors_fn, cost_fn, next_state_diffusion, minimize)


def plot_progress(progress_best, progress_current, temperatures):
    # create two subplots
    # the first subplot shows the best and current costs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(progress_best, label='Best Cost', alpha=0.6)
    plt.plot(progress_current, label='Current Cost', alpha=0.6)
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Simulated Annealing Progress')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(temperatures, label='Temperature', alpha=0.6)
    plt.xlabel('Step')
    plt.ylabel('Temperature')
    plt.title('Temperature Progress')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    @lru_cache(maxsize=2**13)
    def tsp_cost(path: tuple):
        cost = 0

        for i in range(len(path)):
            u = min(path[i], path[(i + 1) % len(path)])
            v = max(path[i], path[(i + 1) % len(path)])
            cost += graph.edges[u, v]['weight']

        return cost


    def tsp_neighbor(path: tuple):
        new_path = list(path).copy()

        i, j = random.sample(range(len(new_path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]

        new_path = tuple(new_path)

        return new_path


    @lru_cache(maxsize=None)
    def tsp_neighbors(path: tuple):
        neighbors = []

        for i, j in combinations(range(len(path)), 2):
            new_path = list(path).copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            new_path = tuple(new_path)
            neighbors.append(new_path)

        return neighbors


    @lru_cache(maxsize=None)
    def neighbors_and_cost(path: tuple):
        neighbors = tsp_neighbors(path)
        costs = np.array([tsp_cost(neighbor) for neighbor in neighbors])

        return neighbors, costs


    def brute_force_tsp(number_of_cities):
        """
        Brute force solution to the TSP problem.
        Returns the optimal path and its cost.
        """
        optimal_cost = float('inf')
        optimal_path = None

        for perm in permutations(range(number_of_cities)):
            cost = tsp_cost(tuple(perm))
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_path = perm

        if optimal_path is None:
            raise ValueError("No optimal path found.")

        return list(optimal_path), optimal_cost


    random.seed(42)
    np.random.seed(42)
    number_of_cities = 100
    graph = nx.complete_graph(number_of_cities)
    pos = nx.spring_layout(graph)
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(1, 100)

    initial_state = tuple(range(number_of_cities))
    t0 = 75.0
    num_steps = 100000
    best_state, best_cost, progress_best, progress_current, temperatures = simulated_annealing(t0, num_steps, initial_state, tsp_neighbor, tsp_cost)
    plot_progress(progress_best, progress_current, temperatures)

    t0 = 50.0
    num_steps = 10000
    best_state, best_cost, progress_best, progress_current, temperatures = simulated_diffusion(t0, num_steps, initial_state, neighbors_and_cost, tsp_cost)
    plot_progress(progress_best, progress_current, temperatures)
