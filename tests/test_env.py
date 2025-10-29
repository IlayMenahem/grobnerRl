from typing import Any, cast
import itertools

import numpy as np
import sympy as sp
from sympy.polys.rings import PolyElement
import pytest

from grobnerRl.envs.env import (
    tokenize,
    make_obs,
    spoly,
    reduce as poly_reduce,
    update,
    minimalize,
    interreduce,
    select,
    buchberger,
    GVW_buchberger,
    BuchbergerEnv,
    GVWEnv,
)
from grobnerRl.envs.ideals import IdealGenerator


R, x, y = sp.ring('x,y', sp.QQ, 'lex')


class DummyIdealGenerator(IdealGenerator):
    def __init__(self, batches):
        super().__init__()
        self._data = iter(batches)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data)


def expected_token(poly):
    coeffs = np.array(list(map(int, poly.coeffs())), dtype=int).reshape((-1, 1))
    monoms = np.array(poly.monoms(), dtype=int)
    return np.concatenate((coeffs, monoms), axis=1)


def test_tokenize_returns_expected_arrays():
    polys = [x + y, x * y + 2]
    tokens = tokenize(polys)
    assert len(tokens) == len(polys)
    for poly, token in zip(polys, tokens):
        assert isinstance(token, np.ndarray)
        assert np.array_equal(token, expected_token(poly))


def test_tokenize_handles_negative_coefficients():
    poly = -x + 2 * y
    tokens = tokenize([poly])
    assert tokens[0].shape == expected_token(poly).shape
    assert np.array_equal(tokens[0], expected_token(poly))


def test_make_obs_tokenizes_and_copies_pairs():
    generators = [x + y]
    pairs = [(0, 1)]
    obs_generators, obs_pairs = make_obs(generators, pairs)
    assert obs_pairs == pairs and obs_pairs is not pairs
    assert np.array_equal(obs_generators[0], expected_token(generators[0]))


def test_make_obs_does_not_mutate_input():
    generators = [x + y]
    pairs = [(0, 1)]
    make_obs(generators, pairs)
    assert generators == [x + y]
    assert pairs == [(0, 1)]


def test_spoly_matches_expected_polynomial():
    f = x**2 + y
    g = x * y + 1
    assert spoly(f, g) == R(y**2 - x)


def test_spoly_returns_zero_for_redundant_pair():
    assert spoly(x, y) == R.zero


def test_reduce_returns_remainder_and_stats():
    remainder, stats = poly_reduce(x * y + 1, [x])
    assert remainder == R.one
    assert stats == {"steps": 1}


def test_reduce_handles_multiple_divisors():
    dividend = x**2 * y + x * y + y
    remainder, stats = poly_reduce(dividend, [x * y + 1, y + 1])
    assert remainder == R(-x - 2)
    assert stats == {"steps": 3}


def test_reduce_leaves_polynomial_when_no_divisor_applies():
    dividend = x + 1
    remainder, stats = poly_reduce(dividend, [y])
    assert remainder == R(dividend)
    assert stats == {"steps": 0}


def test_update_appends_polynomial_and_prunes_pairs():
    generators = [x]
    pairs = []
    update(generators, pairs, y)
    assert generators == [x, y]
    assert pairs == []

    update(generators, pairs, x * y)
    assert generators == [x, y, x * y]
    assert pairs == [(0, 2)]


def test_update_filters_chain_redundant_pairs():
    generators = [x, y**2]
    pairs = [(0, 1)]
    update(generators, pairs, y)
    assert generators == [x, y**2, y]
    assert pairs == [(1, 2)]


def test_minimalize_filters_redundant_generators():
    generators = [x, x * y, y**2, x**2]
    minimal = minimalize(generators)
    assert minimal == [y**2, x]


def test_interreduce_returns_monic_remainders():
    generators = [x + y, y]
    reduced = interreduce(generators)
    assert reduced == [x, y]


def assert_reduced_groebner_basis(basis: list[PolyElement]):
    if not basis:
        return
    assert all(poly.LC == 1 for poly in basis)
    ring = basis[0].ring
    ordered = sorted([poly.copy() for poly in basis], key=lambda f: ring.order(f.LM))
    reduced = interreduce([poly.copy() for poly in ordered])
    assert len(ordered) == len(reduced)
    for original, reduced_poly in zip(ordered, reduced):
        assert original == reduced_poly

    for poly1, poly2 in itertools.combinations(basis, 2):
        s = spoly(poly1, poly2)
        remainder, _ = poly_reduce(s, basis)
        assert remainder == 0


def katsura_system(n: int):
    var_names = ','.join(f'x{i}' for i in range(n + 1))
    R, *vars_ = sp.ring(var_names, sp.QQ, 'lex')
    polynomials = []
    total = sum(vars_[1:-1], R.zero) if n > 1 else R.zero
    polynomials.append(vars_[0] + 2 * total + vars_[-1] - 1)
    for k in range(1, n):
        acc = R.zero
        for i in range(0, n + 1 - k):
            acc += vars_[i] * vars_[i + k]
        polynomials.append(acc - vars_[k])
    polynomials.append(sum((var**2 for var in vars_), R.zero) - vars_[0])
    return R, vars_, polynomials


def katsura_expected_basis(R, vars_, n: int):
    if n == 1:
        return [
            vars_[1]**2 - sp.Rational(1, 2) * vars_[1],
            vars_[0] + vars_[1] - 1,
        ]
    if n == 2:
        return [
            vars_[2]**3 - sp.Rational(1, 2) * vars_[2]**2,
            vars_[1] * vars_[2] - sp.Rational(1, 2) * vars_[1] + sp.Rational(1, 2) * vars_[2]**2 - sp.Rational(1, 4) * vars_[2],
            vars_[1]**2,
            vars_[0] + 2 * vars_[1] + vars_[2] - 1,
        ]
    if n == 3:
        return [
            vars_[3]**5 - sp.Rational(25, 18) * vars_[3]**4 + sp.Rational(4, 9) * vars_[3]**3,
            vars_[2] * vars_[3]**3 - sp.Rational(25, 18) * vars_[2] * vars_[3]**2 + sp.Rational(4, 9) * vars_[2] * vars_[3]
            + sp.Rational(1, 2) * vars_[3]**4 - sp.Rational(25, 36) * vars_[3]**3 + sp.Rational(2, 9) * vars_[3]**2,
            vars_[2]**2 + sp.Rational(45, 8) * vars_[3]**4 - sp.Rational(53, 16) * vars_[3]**3 + sp.Rational(1, 4) * vars_[3]**2,
            vars_[1] + sp.Rational(153, 16) * vars_[2] * vars_[3]**2 - sp.Rational(281, 32) * vars_[2] * vars_[3] + vars_[2]
            - sp.Rational(243, 64) * vars_[3]**4 + sp.Rational(1143, 128) * vars_[3]**3 - sp.Rational(289, 64) * vars_[3]**2
            + sp.Rational(1, 2) * vars_[3],
            vars_[0] - sp.Rational(153, 8) * vars_[2] * vars_[3]**2 + sp.Rational(281, 16) * vars_[2] * vars_[3]
            + sp.Rational(243, 32) * vars_[3]**4 - sp.Rational(1143, 64) * vars_[3]**3 + sp.Rational(289, 32) * vars_[3]**2 - 1,
        ]
    raise ValueError('Unsupported Katsura instance size')


def test_select_supports_various_strategies():
    basis = [x**2 + 1, x * y + 1, y**2 + 1]
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert select(basis, pairs, strategy="normal") == (1, 2)
    assert select(basis, pairs, strategy="degree") == (0, 1)
    combo_strategy = cast(Any, ["degree", "first"])
    assert select(basis, pairs, strategy=combo_strategy) == (0, 1)

    np.random.seed(0)
    assert select(basis, pairs, strategy="random") == (0, 1)

    basis2 = [x**2 + y, x * y + 1, y**2 + x]
    pairs2 = [(0, 1), (0, 2), (1, 2)]
    assert select(basis2, pairs2, strategy="degree_after_reduce") == (0, 2)


def test_select_degree_after_reduce_handles_zero_remainder_pair():
    basis = [x, y, x + y]
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert select(basis, pairs, strategy="degree_after_reduce") == (0, 1)


def test_buchberger_returns_expected_basis_and_stats():
    basis, stats = buchberger([x * y - 1, x - 1])
    assert {g for g in basis} == {x - 1, y - 1}
    assert_reduced_groebner_basis(basis)
    assert stats == {
        "zero_reductions": 0,
        "nonzero_reductions": 1,
        "total_reduction_steps": 0,
        "pairs_processed": 1,
    }


def test_buchberger_nontrivial_ideal_produces_expected_basis():
    basis, stats = buchberger([x**2 - y, x * y - 1])
    assert {g for g in basis} == {x - y**2, y**3 - 1}
    assert_reduced_groebner_basis(basis)
    assert stats["pairs_processed"] == 3
    assert stats["nonzero_reductions"] == 2


def test_buchberger_handles_empty_input():
    basis, stats = buchberger([])
    assert basis == []
    assert_reduced_groebner_basis(basis)
    assert stats == {
        "zero_reductions": 0,
        "nonzero_reductions": 0,
        "total_reduction_steps": 0,
        "pairs_processed": 0,
    }


def test_buchberger_env_reset_eval_mode():
    env = BuchbergerEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    (generators, pairs), info = env.reset()
    assert generators == [x**2 + y, x * y + 1]
    assert pairs == [(0, 1)]
    assert info == {}


def test_buchberger_env_reset_train_mode_tokenizes_state():
    env = BuchbergerEnv(DummyIdealGenerator([[x + y]]), mode="train")
    (tokenized_generators, pairs), info = env.reset()
    assert len(tokenized_generators) == 1
    assert np.array_equal(tokenized_generators[0], expected_token(x + y))
    assert pairs == []
    assert info == {}


def test_buchberger_env_step_with_tuple_action_updates_state():
    env = BuchbergerEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    env.reset()
    (generators, pairs), reward, terminated, truncated, info = env.step((0, 1))
    assert generators == [x**2 + y, x * y + 1, x - y**2]
    assert pairs == [(0, 2), (1, 2)]
    assert reward == -1
    assert terminated is False and truncated is False
    assert info == {}


def test_buchberger_env_step_accepts_integer_actions():
    env = BuchbergerEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    env.reset()
    env.step(1)
    assert env.generators == [x**2 + y, x * y + 1, x - y**2]
    assert env.pairs == [(0, 2), (1, 2)]


def test_buchberger_env_step_zero_remainder_terminates_episode():
    env = BuchbergerEnv(DummyIdealGenerator([[x, y]]), mode="eval")
    env.reset()
    _, reward, terminated, truncated, _ = env.step((0, 1))
    assert reward == -1
    assert terminated is True
    assert truncated is False

def test_gvw_buchberger_simple_ideal():
    basis, syzygies = GVW_buchberger([x * y - 1, x - 1])
    assert {g for g in basis} == {x - 1, y - 1}
    assert_reduced_groebner_basis(basis)
    assert isinstance(syzygies, list)


def test_gvw_buchberger_nontrivial_ideal():
    basis, syzygies = GVW_buchberger([x**2 - y, x * y - 1])
    assert {g for g in basis} == {x - y**2, y**3 - 1}
    assert_reduced_groebner_basis(basis)
    assert isinstance(syzygies, list)


def test_gvw_buchberger_empty_input():
    basis, syzygies = GVW_buchberger([])
    assert basis == []
    assert syzygies == []
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_single_generator():
    basis, syzygies = GVW_buchberger([x + y])
    assert basis == [x + y]
    assert_reduced_groebner_basis(basis)
    assert isinstance(syzygies, list)


def test_gvw_buchberger_already_groebner():
    basis, syzygies = GVW_buchberger([x, y])
    assert {g for g in basis} == {x, y}
    assert_reduced_groebner_basis(basis)
    assert isinstance(syzygies, list)


def test_gvw_buchberger_with_zero_polynomials():
    basis, syzygies = GVW_buchberger([x + y, R.zero, x * y])
    assert R.zero not in basis
    assert len(basis) > 0
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_three_generators():
    basis, syzygies = GVW_buchberger([x**2 + y**2 - 1, x - y, x * y])
    assert len(basis) > 0
    assert all(g != 0 for g in basis)
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_produces_minimal_basis():
    basis, _ = GVW_buchberger([x**2, x * y, y**2])
    minimal = minimalize(basis)
    assert len(basis) == len(minimal)
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_produces_interreduced_basis():
    basis, _ = GVW_buchberger([x**2 + x, x + 1])
    for i, poly in enumerate(basis):
        remainder, _ = poly_reduce(poly, basis[:i] + basis[i+1:])
        assert remainder == poly or len(basis) == 1
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_cyclic_ideal():
    basis, syzygies = GVW_buchberger([x + y, x * y - 1])
    assert len(basis) == 2
    assert all(g.monic() == g for g in basis)
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_homogeneous_ideal():
    basis, syzygies = GVW_buchberger([x**2 - y**2, x * y])
    assert len(basis) > 0
    assert isinstance(syzygies, list)
    assert_reduced_groebner_basis(basis)


def test_gvw_buchberger_returns_monic_basis():
    basis, _ = GVW_buchberger([2*x + 3*y, 5*x*y - 7])
    assert all(g.LC == 1 for g in basis)
    assert_reduced_groebner_basis(basis)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_katsura_systems(n):
    R, vars_, polynomials = katsura_system(n)
    expected_basis = katsura_expected_basis(R, vars_, n)

    buchberger_basis, _ = buchberger(polynomials)
    gvw_basis, _ = GVW_buchberger(polynomials)

    assert {g for g in buchberger_basis} == {g for g in expected_basis}
    assert {g for g in gvw_basis} == {g for g in expected_basis}

    assert_reduced_groebner_basis(buchberger_basis)
    assert_reduced_groebner_basis(gvw_basis)


# Tests adapted from test_buchberger.py

R1, x1, y1, z1 = sp.ring('x,y,z', sp.FF(32003), 'grevlex')
R2, a, b, c, d = sp.ring('a,b,c,d', sp.QQ, 'lex')
R3, t, u, v = sp.ring('t,u,v', sp.FF(101), 'grlex')


@pytest.mark.parametrize("f, g, s", [
    (x1**2 + x1*y1, y1**2 + x1*y1, 0),
    (x1**3*y1**2 - x1**2*y1**3, x1**4*y1 + y1**2, -x1**3*y1**3 - y1**3),
    (x1**2 + y1**3, x1*y1**2 + x1 + 1, x1**3 - x1*y1 - y1),
    (a**2 + a*b, b**2 + a*b, 0),
    (a**3*b**2 - a**2*b**3, a**4*b + b**2, -a**3*b**3 - b**3),
    (a**2 - b**3, a*b**2 + a + 1, -b**5 - a**2 - a),
    (t**2 + t*u, u**2 + t*u, 0),
    (t**3*u**2 - t**2*u**3, t**4*u + u**2, -t**3*u**3 - u**3),
    (t**2 + u**3, t*u**2 + t + 1, t**3 - t*u - u),
])
def test_spoly_from_buchberger(f, g, s):
    assert spoly(f, g) == s


@pytest.mark.parametrize("g, F, r, s", [
    (x1**5*y1**10*z1**4 + 22982*x1**3*y1*z1**2,
     [x1**5*y1**12 + 25797*x1*y1**5*z1**2, x1*y1**3*z1 + 27630*x1**2*y1, x1**2*y1**9*z1 + 8749*x1**2],
     2065*x1**9*y1**2 + 22982*x1**3*y1*z1**2,
     4),
    (a**5*c + a**3*b + a**2*b**2 + a*b**2 + a,
     [a**2*c - a, a*b**2 + c**5, a*c + c**3/4],
     a**4 + a**3*b + a + c**7/4 - c**5,
     4),
    (a**3*b*c**2 + a**2*c,
     [a**2 + b, a*b*c + c, a*c**2 + b**2],
     b*c**2 - b*c,
     3),
])
def test_reduce_from_buchberger(g, F, r, s):
    assert poly_reduce(g, F) == (r, {'steps': s})


@pytest.mark.parametrize("s, p", [
    ('degree', (0, 1)), ('normal', (0, 1)), ('first', (0, 1)),
])
def test_select_buchberger_0(s, p):
    G = [x1**2 + y1, x1*y1 + x1, z1**3 + x1 + y1]
    P = [(0, 1), (0, 2), (1, 2)]
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    (['degree', 'first'], (0, 2)), ('normal', (1, 2)), ('first', (0, 1)),
])
def test_select_buchberger_1(s, p):
    G = [x1*y1 + 1, z1**2 + x1 + z1, y1*z1 + x1]
    P = [(0, 1), (0, 2), (1, 2)]
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('normal', (0, 2)), ('first', (0, 2)), ('random', (0, 2)),
])
def test_select_buchberger_2(s, p):
    G = [x1*y1 + 1, z1**2 + x1 + z1, y1*z1 + x1]
    P = [(0, 2)]
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    (['degree', 'first'], (0, 1)),
    (['degree', 'normal'], (1, 3)),
    ('normal', (1, 2)),
])
def test_select_buchberger_3(s, p):
    G = [a*b + c*d**3, c*d + d, d**5, c**2*d**2]
    P = [(0, 1), (1, 2), (1, 3)]
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('first', (0, 2)), ('normal', (1, 2)),
    (['degree', 'first'], (1, 3)),
    (['degree', 'normal'], (1, 4)),
])
def test_select_buchberger_4(s, p):
    G = [a*b*c, c*d, d**5, a*b, c**2*d**2]
    P = [(0, 2), (1, 2), (1, 3), (1, 4)]
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('first', (1, 2)),
    (['first', 'random'], (1, 2)),
    ('normal', (0, 3)),
    (['degree', 'first'], (0, 3)),
    (['degree', 'normal', 'first'], (0, 3)),
])
def test_select_buchberger_5(s, p):
    G = [t*u**2 + t**2, u*v + 1, v**5 + t, u**3 + t*u]
    P = [(0, 3), (1, 2)]
    assert select(G, P, strategy=s) == p


def test_update_from_buchberger_0():
    # Test with x1 from R1
    G = []
    P = []
    update(G, P, x1**2 + x1*y1 + 2)
    assert G == [x1**2 + x1*y1 + 2]
    assert P == []


def test_update_from_buchberger_1():
    G = [x1*y1**2 + 2*x1*z1 - x1]
    P = []
    f = z1**5 + 2*x1**2*y1*z1 + x1*z1
    update(G, P, f)
    assert G == [x1*y1**2 + 2*x1*z1 - x1, f]
    assert P == []


def test_update_from_buchberger_2():
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    P = [(0, 1)]
    f = a + b**2*c + 4*c**2 + 1
    update(G, P, f)
    assert len(G) == 3 and G[2] == f
    # The update function in env.py uses Gebauer-Moeller criteria by default
    assert (0, 2) in P and (1, 2) in P


def test_update_from_buchberger_3():
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    P = [(0, 1)]
    f = 4*c**2 + 1
    update(G, P, f)
    assert len(G) == 3 and G[2] == f
    assert (0, 2) in P or (1, 2) in P


def test_update_from_buchberger_4():
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    P = [(0, 1)]
    f = 4*b**2*c + b*c**2
    update(G, P, f)
    assert len(G) == 3 and G[2] == f
    # Check that at least some pairs were added
    assert len(P) > 0


@pytest.mark.parametrize("G, Gmin", [
    ([], []),
    ([x1*y1**2 + z1, x1*z1 + 3*y1, x1**2 + y1*z1, -3*y1**3 + z1**2, -3*y1 - z1**3/3, z1**8/243 + z1],
     [x1*z1 + 3*y1, x1**2 + y1*z1, -z1**3/3 - 3*y1, -3*y1**3 + z1**2, x1*y1**2 + z1]),
    ([a*b**2 + c, a*c + 3*b, a**2 + b*c, -3*b**3 + c**2, -3*b - c**3/3, c**8/243 + c],
     [c**8/243 + c, -3*b - c**3/3, a*c + 3*b, a**2 + b*c]),
])
def test_minimalize_from_buchberger(G, Gmin):
    assert minimalize(G) == Gmin


@pytest.mark.parametrize("G, Gred", [
    ([], []),
    ([x1*z1 + 3*y1, x1**2 + y1*z1, -z1**3/3 - 3*y1, -3*y1**3 + z1**2, x1*y1**2 + z1],
     [x1*z1 + 3*y1, x1**2 + y1*z1, z1**3 + 9*y1, y1**3 - z1**2/3, x1*y1**2 + z1]),
    ([c**8/243 + c, -3*b - c**3/3, a*c + 3*b, a**2 + b*c],
     [c**8 + 243*c, b + c**3/9, a*c - c**3/3, a**2 - c**4/9]),
])
def test_interreduce_from_buchberger(G, Gred):
    assert interreduce(G) == Gred


@pytest.mark.parametrize("F, G", [
    ([], []),
    ([y1 - x1**2, z1 - x1**3], [y1**2 - x1*z1, x1*y1 - z1, x1**2 - y1]),
    ([b - a**2, c - a**3], [b**3 - c**2, a*c - b**2, a*b - c, a**2 - b]),
    ([u - t**2, v - t**3], [t*v - u**2, t*u - v, t**2 - u, u**3 - v**2]),
    ([x1 + y1 + z1, x1*y1 + y1*z1 + x1*z1, x1*y1*z1 - 1], [x1 + y1 + z1, y1**2 + y1*z1 + z1**2, z1**3 - 1]),
])
def test_buchberger_from_buchberger(F, G):
    result, _ = GVW_buchberger(F)
    assert result == G
    result2, _ = buchberger(F)
    assert result2 == G


# ===========================
# GVWEnv Tests (Challenging)
# ===========================


def test_gvw_env_reset_eval_mode_initializes_signatures():
    """Test that GVWEnv properly initializes with signature pairs."""
    env = GVWEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    (generators, pairs), info = env.reset()
    
    # Should have initialized with signature pairs
    assert len(pairs) == 2
    assert len(generators) == 0  # No generators yet until pairs are processed
    assert info == {}
    
    # Each pair should be a signature tuple: ((monomial_tuple, int))
    for pair in pairs:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(pair[0], tuple)  # monomial
        assert isinstance(pair[1], int)    # index


def test_gvw_env_reset_train_mode_tokenizes_state():
    """Test that GVWEnv tokenizes observation in train mode."""
    env = GVWEnv(DummyIdealGenerator([[x + y]]), mode="train")
    (tokenized_generators, pairs), info = env.reset()
    
    assert isinstance(tokenized_generators, list)
    assert isinstance(pairs, list)
    assert info == {}


def test_gvw_env_step_with_integer_action_processes_pair():
    """Test stepping with integer action index."""
    env = GVWEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    env.reset()
    
    initial_pair_count = len(env.pairs)
    assert initial_pair_count > 0
    
    # Step with first pair
    (generators, pairs), reward, terminated, truncated, info = env.step(0)
    
    assert reward == -1
    assert truncated is False
    assert isinstance(generators, list)
    assert isinstance(pairs, list)


def test_gvw_env_step_with_signature_action_processes_correct_pair():
    """Test stepping with explicit signature tuple."""
    env = GVWEnv(DummyIdealGenerator([[x**2 - y, x * y - 1]]), mode="eval")
    env.reset()
    
    # Get the first signature pair
    signature = env.pairs[0]
    
    # Step using the signature
    (generators, pairs), reward, terminated, truncated, info = env.step(signature)
    
    assert reward == -1
    assert not truncated
    # The signature we used should no longer be in pairs (it was processed)
    assert signature not in pairs


def test_gvw_env_step_with_invalid_signature_raises_error():
    """Test that invalid signature raises ValueError."""
    env = GVWEnv(DummyIdealGenerator([[x + y]]), mode="eval")
    env.reset()
    
    # Create a fake signature that doesn't exist
    fake_signature = ((999, 999), 999)
    
    with pytest.raises(ValueError, match="signature not found"):
        env.step(fake_signature)


def test_gvw_env_step_with_invalid_action_type_raises_error():
    """Test that invalid action type raises TypeError."""
    env = GVWEnv(DummyIdealGenerator([[x + y]]), mode="eval")
    env.reset()
    
    # Try with a plain tuple of ints (should raise TypeError)
    with pytest.raises(TypeError, match="action must be an index or signature tuple"):
        env.step((0, 1))


def test_gvw_env_step_with_out_of_bounds_index_raises_error():
    """Test that out-of-bounds index raises IndexError."""
    env = GVWEnv(DummyIdealGenerator([[x + y]]), mode="eval")
    env.reset()
    
    with pytest.raises(IndexError, match="action index out of range"):
        env.step(999)
    
    with pytest.raises(IndexError, match="action index out of range"):
        env.step(-1)


def test_gvw_env_step_when_no_pairs_raises_error():
    """Test that stepping when no pairs available raises ValueError."""
    env = GVWEnv(DummyIdealGenerator([[x]]), mode="eval")
    env.reset()
    
    # Process all pairs until termination
    while env.pairs:
        env.step(0)
    
    # Now try to step again
    with pytest.raises(ValueError, match="no pairs available to process"):
        env.step(0)


def test_gvw_env_terminates_when_all_pairs_processed():
    """Test that environment terminates correctly."""
    env = GVWEnv(DummyIdealGenerator([[x * y - 1, x - 1]]), mode="eval")
    env.reset()
    
    terminated = False
    step_count = 0
    max_steps = 100  # Safety limit
    final_pairs = []
    
    while not terminated and step_count < max_steps:
        if not env.pairs:
            break
        (generators, final_pairs), reward, terminated, truncated, _ = env.step(0)
        step_count += 1
    
    assert terminated or len(final_pairs) == 0
    assert step_count < max_steps  # Should finish in reasonable time


def test_gvw_env_produces_valid_groebner_basis():
    """Test that GVWEnv produces a valid Groebner basis."""
    env = GVWEnv(DummyIdealGenerator([[x**2 - y, x * y - 1]]), mode="eval")
    env.reset()
    
    # Process all pairs
    while env.pairs:
        env.step(0)
    
    # Check that final generators form a Groebner basis
    # Note: GVWEnv doesn't automatically minimize/interreduce, so we do it manually
    final_generators = env.generators
    minimal_basis = minimalize(final_generators)
    reduced_basis = interreduce(minimal_basis) if minimal_basis else []
    assert_reduced_groebner_basis(reduced_basis)


def test_gvw_env_complex_ideal_correct_result():
    """Test GVWEnv on a more complex ideal."""
    polynomials = [x**2 + y**2 - 1, x - y, x * y]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    # Process all pairs
    step_count = 0
    max_steps = 200
    while env.pairs and step_count < max_steps:
        env.step(0)
        step_count += 1
    
    # Verify basis is valid
    assert len(env.generators) > 0
    assert all(g != 0 for g in env.generators)
    minimal_basis = minimalize(env.generators)
    reduced_basis = interreduce(minimal_basis) if minimal_basis else []
    assert_reduced_groebner_basis(reduced_basis)


def test_gvw_env_state_consistency_across_steps():
    """Test that internal state remains consistent across steps."""
    env = GVWEnv(DummyIdealGenerator([[x**2 - y, x * y - 1]]), mode="eval")
    env.reset()
    
    while env.pairs:
        # Check state consistency before step
        assert len(env._state['jpairs']) == len(env.pairs)
        assert env.generators == env._state['generators']
        
        env.step(0)
        
        # Check state consistency after step
        assert len(env._state['jpairs']) == len(env.pairs)
        assert env.generators == env._state['generators']


def test_gvw_env_handles_zero_polynomials():
    """Test that GVWEnv handles zero polynomials in input."""
    env = GVWEnv(DummyIdealGenerator([[x + y, R.zero, x * y]]), mode="eval")
    (generators, pairs), _ = env.reset()
    
    # Process all pairs
    while env.pairs:
        env.step(0)
    
    # Zero should not appear in final generators
    assert R.zero not in env.generators
    assert len(env.generators) > 0


def test_gvw_env_empty_ideal():
    """Test GVWEnv with empty ideal."""
    env = GVWEnv(DummyIdealGenerator([[]]), mode="eval")
    (generators, pairs), info = env.reset()
    
    assert generators == []
    assert pairs == []
    assert info == {}


def test_gvw_env_single_generator():
    """Test GVWEnv with single polynomial."""
    env = GVWEnv(DummyIdealGenerator([[x + y]]), mode="eval")
    (generators, pairs), _ = env.reset()
    
    assert len(pairs) == 1  # One signature pair for single generator
    
    # Process the pair
    while env.pairs:
        env.step(0)
    
    # Should have one generator in final basis
    assert len(env.generators) == 1


def test_gvw_env_already_groebner_basis():
    """Test GVWEnv when input is already a Groebner basis."""
    env = GVWEnv(DummyIdealGenerator([[x, y]]), mode="eval")
    env.reset()
    
    # Process all pairs
    while env.pairs:
        env.step(0)
    
    # Should produce the same basis (or equivalent)
    assert len(env.generators) == 2
    assert_reduced_groebner_basis(env.generators)


def test_gvw_env_different_action_orders_produce_same_basis():
    """Test that different action selection orders produce equivalent bases."""
    polynomials = [x**2 - y, x * y - 1]
    
    # First run: always select first pair
    env1 = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env1.reset()
    while env1.pairs:
        env1.step(0)
    reduced_basis1 = interreduce(minimalize(env1.generators))
    basis1 = set(reduced_basis1)
    
    # Second run: always select last pair
    env2 = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env2.reset()
    while env2.pairs:
        env2.step(len(env2.pairs) - 1)
    reduced_basis2 = interreduce(minimalize(env2.generators))
    basis2 = set(reduced_basis2)
    
    # Both should produce valid Groebner bases
    assert_reduced_groebner_basis(list(basis1))
    assert_reduced_groebner_basis(list(basis2))
    
    # They should be equivalent (same ideal)
    assert basis1 == basis2


def test_gvw_env_matches_gvw_buchberger_result():
    """Test that GVWEnv produces the same result as GVW_buchberger."""
    polynomials = [x**2 - y, x * y - 1]
    
    # Compute basis using GVW_buchberger
    expected_basis, _ = GVW_buchberger(polynomials)
    
    # Compute basis using GVWEnv
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    while env.pairs:
        env.step(0)
    
    # Reduce the env basis to match GVW_buchberger output
    env_basis = interreduce(minimalize(env.generators)) if env.generators else []
    
    # Should match
    assert set(env_basis) == set(expected_basis)


def test_gvw_env_reward_is_always_negative_one():
    """Test that reward is always -1 for each step."""
    env = GVWEnv(DummyIdealGenerator([[x**2 - y, x * y - 1]]), mode="eval")
    env.reset()
    
    rewards = []
    while env.pairs:
        _, reward, _, _, _ = env.step(0)
        rewards.append(reward)
    
    assert all(r == -1 for r in rewards)


def test_gvw_env_multiple_resets():
    """Test that environment can be reset multiple times."""
    ideal1 = [x + y]
    ideal2 = [x**2 - y, x * y - 1]
    
    env = GVWEnv(DummyIdealGenerator([ideal1, ideal2]), mode="eval")
    
    # First episode
    (gen1, pairs1), _ = env.reset()
    initial_pairs_count1 = len(pairs1)
    
    # Second episode
    (gen2, pairs2), _ = env.reset()
    initial_pairs_count2 = len(pairs2)
    
    # Second episode should have different initial state
    assert initial_pairs_count1 != initial_pairs_count2


def test_gvw_env_katsura_systems():
    """Test GVWEnv on challenging Katsura systems."""
    for n in [1, 2]:  # Keep n small for reasonable test time
        R_kat, vars_kat, polynomials = katsura_system(n)
        expected_basis = katsura_expected_basis(R_kat, vars_kat, n)
        
        env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
        env.reset()
        
        # Process all pairs
        step_count = 0
        max_steps = 500
        while env.pairs and step_count < max_steps:
            env.step(0)
            step_count += 1
        
        # Check we got the correct basis
        assert set(env.generators) == set(expected_basis)
        assert_reduced_groebner_basis(env.generators)


def test_gvw_env_noncommutative_lcm_pairs():
    """Test GVWEnv handles pairs with complex LCM relationships."""
    # This creates a scenario where multiple pairs have overlapping LCMs
    polynomials = [x**3 + y**2, x*y**2 + x, x**2*y + y**2]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    # Should handle all pairs without error
    step_count = 0
    max_steps = 300
    while env.pairs and step_count < max_steps:
        env.step(0)
        step_count += 1
    
    assert step_count < max_steps
    reduced_basis = interreduce(minimalize(env.generators)) if env.generators else []
    assert_reduced_groebner_basis(reduced_basis)


def test_gvw_env_homogeneous_ideal():
    """Test GVWEnv on homogeneous ideals."""
    polynomials = [x**2 - y**2, x*y]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    while env.pairs:
        env.step(0)
    
    assert len(env.generators) > 0
    assert_reduced_groebner_basis(env.generators)


def test_gvw_env_cyclic_ideal():
    """Test GVWEnv on cyclic-type ideal."""
    polynomials = [x + y, x * y - 1]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    while env.pairs:
        env.step(0)
    
    # Should produce a minimal basis
    assert len(env.generators) == 2
    assert all(g.monic() == g for g in env.generators)
    assert_reduced_groebner_basis(env.generators)


def test_gvw_env_signature_updates_correctly():
    """Test that signatures are updated correctly during computation."""
    polynomials = [x**2 - y, x * y - 1]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    initial_syzygies = len(env._state.get('syzygies', set()))
    
    # Process some pairs
    steps = min(3, len(env.pairs))
    for _ in range(steps):
        if env.pairs:
            env.step(0)
    
    # Syzygies may have been discovered
    final_syzygies = len(env._state.get('syzygies', set()))
    assert final_syzygies >= initial_syzygies


def test_gvw_env_train_mode_tokenizes_correctly():
    """Test that train mode produces properly tokenized observations."""
    polynomials = [x**2 + y, x * y + 1]
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="train")
    (tokenized_gen, pairs), _ = env.reset()
    
    assert isinstance(tokenized_gen, list)
    # In train mode, generators should be empty initially
    assert len(tokenized_gen) == 0
    
    # Step and check tokenization
    if pairs:
        (tokenized_gen, pairs), _, _, _, _ = env.step(0)
        if len(tokenized_gen) > 0:
            # Should be numpy arrays
            assert all(isinstance(token, np.ndarray) for token in tokenized_gen)


def test_gvw_env_large_coefficient_ideal():
    """Test GVWEnv with polynomials having large coefficients."""
    R_large, x_l, y_l = sp.ring('x,y', sp.QQ, 'lex')
    polynomials = [1000*x_l**2 - 500*y_l, 2000*x_l*y_l - 1000]
    
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    while env.pairs:
        env.step(0)
    
    # Should still produce monic basis
    assert all(g.LC == 1 for g in env.generators)
    reduced_basis = interreduce(minimalize(env.generators)) if env.generators else []
    assert_reduced_groebner_basis(reduced_basis)


def test_gvw_env_with_field_characteristic():
    """Test GVWEnv with finite field."""
    R_ff, x_ff, y_ff, z_ff = sp.ring('x,y,z', sp.FF(32003), 'grevlex')
    polynomials = [x_ff**2 + x_ff*y_ff, y_ff**2 + x_ff*y_ff, z_ff**3 + x_ff]
    
    env = GVWEnv(DummyIdealGenerator([polynomials]), mode="eval")
    env.reset()
    
    step_count = 0
    max_steps = 200
    while env.pairs and step_count < max_steps:
        env.step(0)
        step_count += 1
    
    assert step_count < max_steps
    assert len(env.generators) > 0
    reduced_basis = interreduce(minimalize(env.generators)) if env.generators else []
    assert_reduced_groebner_basis(reduced_basis)