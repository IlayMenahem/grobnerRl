from sympy.polys.rings import PolyElement
from grobnerRl.envs.deepgroebner import reduce, update, interreduce, minimalize, select, spoly


def buchberger(ideal: list[PolyElement]):
    reductions = []
    pairs, basis = init(ideal)

    while pairs:
        selection = select(basis, pairs)
        reductions.append(selection)
        basis, pairs = step(basis, pairs, selection)

    basis = interreduce(minimalize(basis))

    return basis, reductions


def step(basis: list[PolyElement], pairs: list[tuple[int, int]], selection: tuple[int, int]) -> tuple[list[PolyElement], list[tuple[int, int]]]:
    i, j = selection
    pairs.remove((i, j))
    s = spoly(basis[i], basis[j])
    r, _ = reduce(s, basis)

    if r != 0:
        basis, pairs = update(basis, pairs, r.monic())

    return basis, pairs


def init(ideal: list[PolyElement]) -> tuple[list[tuple[int, int]], list[PolyElement]]:
    basis = []
    pairs = []

    for f in ideal:
        basis, pairs = update(basis, pairs, f.monic())

    return pairs, basis


def do_buchberger(ideal: list[PolyElement], selections: list[tuple[int, int]]) -> tuple[list[PolyElement], list[tuple[int, int]]]:
    '''
    does the process of Buchberger's algorithm with the given ideal and selections.

    Args:
    - ideal (list[PolyElement]): The ideal for which the process is displayed.
    - selections (list[tuple[int, int]]): The selections made during the process.

    Returns:
    - (basis, pairs): A tuple containing the basis and pairs after doing Buchberger's
    algorithm with the selections.
    '''
    pairs, basis = init(ideal)

    for selection in selections:
        basis, pairs = step(basis, pairs, selection)

    return basis, pairs
