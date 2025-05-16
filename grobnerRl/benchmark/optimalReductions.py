'''
Optimal or near-optimal reductions for the Groebner basis computation, for small ideals
'''

from collections import deque
from sympy.polys.groebnertools import is_groebner, is_minimal, is_reduced
from sympy.polys.domains import ZZ
from sympy.polys.rings import PolyElement
from grobnerRl.Buchberger.BuchbergerSympy import init, buchberger_step
from grobnerRl.envs.deepgroebner import minimalize, interreduce, buchberger
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

    groebner, _ = buchberger(ideal)
    groebner_lt = set(p.LT for p in groebner)
    def heuristic(basis: list[PolyElement]) -> int:
        '''
        an addmissable heuristic function, the heuristic is the number of leading terms
        in the minimal groebner basis that are not in the current basis.

        Args:
            basis (list): The current basis.
            pairs (list): The current pairs.
            *args: Additional arguments.

        Returns:
            int: The heuristic value.
        '''
        basis_lt = set(p.LT for p in basis)
        remaining_lt = len(groebner_lt - basis_lt)

        return remaining_lt

    pairs, basis = init(ideal)

    best_sequence = None
    best_basis = None
    best_length = float('inf')
    queue: deque[tuple] = deque([(basis, pairs, [])])
    visited = set()

    while queue:
        basis_curr, pairs_curr, seq = queue.popleft()

        # If no pairs left, Grobner basis found
        if not pairs_curr and len(seq) < best_length:
            best_length = len(seq)
            best_sequence = seq
            best_basis = basis_curr

        if len(seq) >= bound:
            return best_sequence, best_basis

        key = state_key(basis_curr, pairs_curr)
        if key in visited:
            continue
        visited.add(key)

        for selection in list(pairs_curr):
            new_basis, new_pairs = buchberger_step(basis_curr.copy(), pairs_curr.copy(), selection)
            new_seq = seq + [selection]
            queue.append((new_basis, new_pairs, new_seq))

    best_basis = interreduce(minimalize(best_basis))

    if not is_groebner(best_basis, ideal[0].ring):
        raise ValueError(f"The basis is not Groebner {best_basis}, the basis is {basis}")

    return best_sequence, best_basis


def test_hypothesis(num_episodes: int, bound: int, *ideal_params) -> bool:
    '''
    test an hypothesis for the optimal reductions

    Args:
        num_episodes (int): The number of episodes to run.
        bound (int): The maximal allowed number of reduction steps.
        ideal_params: [num_polys, max_num_monoms, max_degree, num_vars, field, order]

    Returns:
        bool: True if the hypothesis isn't contradicted, False otherwise.
    '''
    num_success = 0

    for _ in range(num_episodes):
        ideal = random_ideal(*ideal_params)
        basis, _ = buchberger(ideal)
        reductions, basis_opt = optimal_reductions(ideal, bound)

        if not basis_opt:
            continue

        num_success += 1
        print('ideal', len(ideal))
        print('basis', len(basis_opt))
        print('reductions', len(reductions))
        print()

    print(f"Success rate: {num_success/num_episodes:.2f}")

    return True
