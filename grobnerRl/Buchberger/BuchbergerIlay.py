from sympy.polys.rings import PolyElement
from grobnerRl.envs.deepgroebner import reduce, update, interreduce, minimalize, select

def buchberger(ideal: list[PolyElement]):
    pairs, basis = init(ideal)

    while pairs:
        selection = select(basis, pairs)
        basis, pairs = step(basis, pairs, selection)

    basis = interreduce(minimalize(basis))

    return basis


def step(basis: list[PolyElement], pairs: list[tuple[int, int]], selection: tuple[int, int]) -> tuple[list[PolyElement], list[tuple[int, int]]]:
    from grobnerRl.envs.deepgroebner import spoly
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
