import math
import random

def get_temp(T0, k, type='linear'):
    if type == 'log':
        return T0/math.log(k + 2)
    elif type == 'linear':
        return T0/(k + 1)

    raise ValueError(f"Unknown temperature type: {type}. Supported types are 'log' and 'linear'.")


def simulated_annealing(t0: float, num_steps: int, state0, neighbor_fn, cost_fn):
    '''
    Perform simulated annealing to find a near-optimal solution.

    Parameters:
    - t0 (float): Initial temperature.
    - num_steps (int): Number of steps to perform.
    - state0: Initial state.
    - neighbor_fn (function): Function to generate a neighbor state.
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

        neighbor_state = neighbor_fn(current_state)
        neighbor_cost = cost_fn(neighbor_state)

        delta_cost = neighbor_cost - current_cost

        if delta_cost > 0:
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


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    # setup the TSP problem
    random.seed(42)
    number_of_cities = 100
    graph = nx.complete_graph(number_of_cities)
    pos = nx.spring_layout(graph)
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(1, 10)

    def tsp_cost(path: list):
        cost = 0

        for i in range(len(path)):
            u = min(path[i], path[(i + 1) % len(path)])
            v = max(path[i], path[(i + 1) % len(path)])
            cost += graph.edges[u, v]['weight']

        return cost

    def tsp_neighbor(path: list):
        new_path = path.copy()

        i, j = random.sample(range(len(new_path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]

        return new_path

    initial_state = list(range(number_of_cities))
    t0 = 1.0
    num_steps = 100000
    best_state, best_cost, progress_best, progress_current = simulated_annealing(
        t0, num_steps, initial_state, tsp_neighbor, tsp_cost
    )

    def brute_force_tsp(number_of_cities):
        """
        Brute force solution to the TSP problem.
        Returns the optimal path and its cost.
        """
        from itertools import permutations
        optimal_cost = float('inf')
        optimal_path = None

        for perm in permutations(range(number_of_cities)):
            cost = tsp_cost(list(perm))
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_path = perm

        if optimal_path is None:
            raise ValueError("No optimal path found.")

        return list(optimal_path), optimal_cost

    print("Best state:", best_state)
    print("Best cost:", best_cost)

    plt.plot(progress_best, label='Best Cost')
    plt.plot(progress_current, label='Current Cost')
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Simulated Annealing Progress')
    plt.legend()
    plt.show()
