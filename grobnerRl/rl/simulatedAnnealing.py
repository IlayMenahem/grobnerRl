import math
import random
import numpy as np
from functools import lru_cache
from scipy.special import softmax
from itertools import combinations, permutations
import networkx as nx
import matplotlib.pyplot as plt


def get_temp(T0, k, type='linear'):
    if type == 'log':
        return T0/math.log(k + 2)
    elif type == 'linear':
        return T0/(k + 1)

    raise ValueError(f"Unknown temperature type: {type}. Supported types are 'log' and 'linear'.")


def simulated_annealing(t0: float, num_steps: int, state0, neighbor_fn, cost_fn, minimize=True):
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

    for k in range(num_steps):
        temperature = get_temp(t0, k)

        neighbor_state = neighbor_fn(current_state)
        neighbor_cost = cost_fn(neighbor_state)

        delta_cost = neighbor_cost - current_cost

        if (delta_cost > 0 and minimize) or (delta_cost < 0 and not minimize):
            probability = math.exp(-delta_cost / temperature)
        else:
            probability = 1.0

        if random.random() < probability:
            current_state = neighbor_state
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best_state = current_state
            best_cost = current_cost

        progress_best.append(best_cost)
        progress_current.append(current_cost)

    return best_state, best_cost, progress_best, progress_current


def an_simulated_annealing(t0: float, num_steps: int, state0, neighbors_and_costs, cost_fn):
    '''
    Perform simulated annealing to find a near-optimal solution, this method looks at all
    neighbors and their costs at once and chooses one based on the softmax probability.

    it's recommended to use @lru_cache on neighbors_and_costs to speed up the process.

    Parameters:
    - t0 (float): Initial temperature.
    - num_steps (int): Number of steps to perform.
    - state0: Initial state.
    - neighbors_and_costs (function): Function that returns a tuple of neighbors and their costs.
    - cost_fn (function): Function to compute the cost of a state.
    '''
    current_state = state0
    current_cost = cost_fn(current_state)
    best_state = current_state
    best_cost = current_cost

    progress_best = [current_cost]
    progress_current = [current_cost]

    for k in range(num_steps):
        temperature = get_temp(t0, k)

        neighbors_state, neighbors_cost = neighbors_and_costs(current_state)

        probabilities = softmax(current_cost - neighbors_cost / temperature)
        chosen_index = np.random.choice(len(neighbors_state), p=probabilities)

        current_state = neighbors_state[chosen_index]
        current_cost = neighbors_cost[chosen_index]

        if current_cost < best_cost:
            best_state = current_state
            best_cost = current_cost

        progress_best.append(best_cost)
        progress_current.append(current_cost)

    return best_state, best_cost, progress_best, progress_current


if __name__ == "__main__":
    @lru_cache(maxsize=None)
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
    def nighbors_and_cost(path: tuple):
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

    # setup the TSP problem
    random.seed(42)
    np.random.seed(42)
    number_of_cities = 100
    graph = nx.complete_graph(number_of_cities)
    pos = nx.spring_layout(graph)
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(1, 100)

    initial_state = tuple(range(number_of_cities))
    t0 = 1.0
    num_steps = 100000
    best_state, best_cost, progress_best_sa, progress_current_sa = simulated_annealing(
        t0, num_steps, initial_state, tsp_neighbor, tsp_cost
    )

    print("Best state:", best_state)
    print("Best cost:", best_cost)

    t0 = 0.5
    num_steps = 100000
    best_state, best_cost, progress_best, progress_current = an_simulated_annealing(
        t0, num_steps, initial_state, nighbors_and_cost, tsp_cost
    )
    print("Best state (AN):", best_state)
    print("Best cost (AN):", best_cost)

    print(nighbors_and_cost.cache_info())

    # Plot both on the same graph
    plt.figure(figsize=(12, 6))
    plt.plot(progress_best_sa, label='Best Cost (SA)', alpha=0.8)
    plt.plot(progress_current_sa, label='Current Cost (SA)', alpha=0.8)
    plt.plot(progress_best, label='Best Cost (AN)', alpha=0.8)
    plt.plot(progress_current, label='Current Cost (AN)', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Simulated Annealing vs AN Simulated Annealing Progress')
    plt.legend()
    plt.show()
