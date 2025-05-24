'''
Optimal or near-optimal reductions for the Groebner basis computation, for small ideals
'''

import heapq
from collections import deque
from sympy.polys.groebnertools import is_groebner
from sympy.polys.rings import PolyElement
from grobnerRl.Buchberger.BuchbergerIlay import init, step, interreduce, minimalize
from grobnerRl.Buchberger.BuchbergerSympy import groebner
from grobnerRl.envs.ideals import random_ideal

def optimal_reductions(ideal: list, bound: int):
    '''
    Computes the optimal reductions for a given ideal to find the computation that uses the least number of reductions.
    This function uses exhaustive search bounded by `bound`.

    Args:
        ideal (list): The ideal for which to compute the reductions.
        bound (int): The maximal allowed number of reduction steps.

    Returns:
        list: A list of selected pairs representing the reduction sequence.
        list: The Grobner basis of the ideal obtained by the sequence.
    '''
    def state_key(basis: list, pairs: list) -> tuple:
        '''
        Create a unique key for the state based on the basis and pairs.

        Args:
            basis (list): The current basis.
            pairs (list): The current pairs.

        Returns:
            tuple: A tuple representing the state key.
        '''
        basis_key = tuple(sorted(str(p) for p in basis))
        pairs_key = tuple(sorted(pairs))
        return (basis_key, pairs_key)

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

    heap = []
    initial_h = heuristic(basis)
    heapq.heappush(heap, (initial_h, 0, basis, pairs, []))
    visited = {}

    best_sequence = None
    best_basis = None
    best_length = float('inf')

    while heap:
        f, g, basis_curr, pairs_curr, seq = heapq.heappop(heap)

        key = state_key(basis_curr, pairs_curr)
        if key in visited and visited[key] <= g:
            continue
        visited[key] = g

        if not pairs_curr and g < best_length:
            best_length = g
            best_sequence = seq
            best_basis = basis_curr
            break

        if g >= bound:
            continue

        for selection in list(pairs_curr):
            new_basis, new_pairs = step(basis_curr.copy(), pairs_curr.copy(), selection)
            new_seq = seq + [selection]
            new_g = g + 1
            new_h = heuristic(new_basis)
            new_f = new_g + new_h

            heapq.heappush(heap, (new_f, new_g, new_basis, new_pairs, new_seq))

    if not best_sequence:
        return None, None

    best_basis = interreduce(minimalize(best_basis))

    if not is_groebner(best_basis, ring):
        raise ValueError(f"The basis is not Groebner {best_basis}, the basis is {groebner_basis}")

    return best_sequence, best_basis


def experiment(num_episodes: int, bound: int, *ideal_params) -> bool:
    '''
    Run an experiment to test the optimal reductions.

    Args:
        num_episodes (int): The number of episodes to run.
        bound (int): The maximal allowed number of reduction steps.
        ideal_params: [num_polys, max_num_monoms, max_degree, num_vars, field, order]

    Returns:
        bool: True if the hypothesis isn't contradicted, False otherwise.
    '''
    num_success = 0

    for episode in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        basis = groebner(ideal, ideal[0].ring)
        reductions, basis_opt = optimal_reductions(ideal, bound)

        print()
        print('epidose', episode)
        print('ideal', len(ideal))
        print('basis', len(basis))

        if not basis_opt:
            print('fucking shit')
            continue

        print('reductions', len(reductions))

        num_success += 1

    print(f"Success rate: {num_success/num_episodes:.2f}")

    return True
