from typing import Any, cast

import numpy as np
import sympy as sp

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
    BuchbergerEnv,
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
    assert stats == {
        "zero_reductions": 0,
        "nonzero_reductions": 1,
        "total_reduction_steps": 0,
        "pairs_processed": 1,
    }


def test_buchberger_nontrivial_ideal_produces_expected_basis():
    basis, stats = buchberger([x**2 - y, x * y - 1])
    assert {g for g in basis} == {x - y**2, y**3 - 1}
    assert stats["pairs_processed"] == 3
    assert stats["nonzero_reductions"] == 2


def test_buchberger_handles_empty_input():
    basis, stats = buchberger([])
    assert basis == []
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
