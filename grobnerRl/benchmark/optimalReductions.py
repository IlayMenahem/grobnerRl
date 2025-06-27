'''
Optimal or near-optimal reductions for the Groebner basis computation, for small ideals
'''

import heapq
from sympy.polys.rings import PolyElement
from sympy.polys.groebnertools import is_groebner
from grobnerRl.Buchberger.BuchbergerIlay import init, step, interreduce, minimalize
from grobnerRl.Buchberger.BuchbergerSympy import groebner
from grobnerRl.envs.ideals import random_ideal


def state_key(basis: list, pairs: list) -> tuple:
    '''
    Create a unique key for the state based on the basis and pairs.

    Args:
    - basis (list): The current basis.
    - pairs (list): The current pairs.

    Returns:
        tuple: A tuple representing the state key.
    '''
    polys = ((tuple(p.monoms()), tuple(map(int, p.coeffs()))) for p in basis)
    basis_key = tuple(sorted(polys))
    pairs_key = tuple(sorted(pairs))

    return (basis_key, pairs_key)


def optimal_reductions(ideal: list, step_limit: int):
    '''
    Computes the optimal reductions for a given ideal to find the computation that uses the least number of reductions.
    This function uses exhaustive search bounded by `bound`.

    Args:
        ideal (list): The ideal for which to compute the reductions.
        step_limit (int): The maximal allowed number of reduction steps.

    Returns:
        list: A list of selected pairs representing the reduction sequence.
        list: The Grobner basis of the ideal obtained by the sequence.
    '''

    ring = ideal[0].ring
    groebner_basis = groebner(ideal, ring)
    groebner_lt = set(p.LT for p in groebner_basis)
    def heuristic(basis: list[PolyElement]) -> int:
        '''
        an addmissable heuristic function, the heuristic is the number of leading terms
        in the minimal groebner basis that are not in the current basis.

        Args:
            basis (list): The current basis.

        Returns:
            int: The heuristic value.
        '''
        basis_lt = set(p.LT for p in basis)
        remaining_lt = len(groebner_lt - basis_lt)

        return int(remaining_lt)

    pairs, basis = init(ideal)

    visited = {}
    heap = []
    counter = 0  # Add counter for tie-breaking
    initial_h = heuristic(basis)
    heapq.heappush(heap, (initial_h, 0, counter, basis, pairs, []))
    counter += 1

    num_steps = 0
    best_sequence = None
    best_basis = None
    best_length = float('inf')

    while heap:
        f, g, _, basis_curr, pairs_curr, seq = heapq.heappop(heap)

        key = state_key(basis_curr, pairs_curr)
        if key in visited and visited[key] <= g:
            continue
        visited[key] = g

        if not pairs_curr and g < best_length:
            best_length = g
            best_sequence = seq
            best_basis = basis_curr
            break

        if num_steps >= step_limit:
            break

        for selection in list(pairs_curr):
            num_steps += 1
            new_basis, new_pairs = step(basis_curr.copy(), pairs_curr.copy(), selection)
            new_seq = seq + [selection]
            new_g = g + 1
            new_h = heuristic(new_basis)
            new_f = new_g + new_h

            heapq.heappush(heap, (new_f, new_g, counter, new_basis, new_pairs, new_seq))
            counter += 1

    if not best_sequence:
        return None, None, num_steps

    best_basis = interreduce(minimalize(best_basis))

    if not is_groebner(best_basis, ideal[0].ring):
        raise ValueError("The computed basis is not a Groebner basis.")

    return best_sequence, best_basis, num_steps


def neighbor(ideal: list, reductions: list) -> list:
    raise NotImplementedError("This function should be implemented to generate a neighbor state based on the current reductions.")


def experiment(num_episodes: int, step_limit: int, *ideal_params) -> float:
    '''
    Run an experiment to test the optimal reductions.

    Args:
        num_episodes (int): The number of episodes to run.
        step_limit (int): The maximum number of reduction steps allowed.
        ideal_params: [num_polys, max_num_monoms, max_degree, num_vars, field, order]

    Returns:
        float: success rate of the experiment.
    '''
    num_success = 0

    for episode in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        reductions, basis, num_steps = optimal_reductions(ideal, step_limit)

        print()
        print('epidose', episode)
        print('num_steps', num_steps)

        if not basis:
            print('רע')
            continue

        print('basis', len(basis))
        print('reductions', len(reductions))

        num_success += 1

    succsess_rate = num_success/num_episodes

    return succsess_rate
