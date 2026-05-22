"""Tests for grobnerRl.experts."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from grobnerRl.env import BuchbergerEnv
from grobnerRl.experts import (
    BasicExpert,
    ClosestLMExpert,
    LeastRemainingPairsExpert,
    LowestLMExpert,
    MCTSExpert,
    OptimalDPExpert,
    RolloutExpert,
    get_basis,
    get_leading_terms,
    lm_by_pair,
    next_step,
    select,
)
from tests.conftest import DummyIdealGenerator, assert_reduced_groebner_basis


def _fresh_env(simple_ideal) -> BuchbergerEnv:
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal]), mode="eval")
    env.reset()
    return env


# -------- select helper --------


def test_select_helper_unknown_strategy_raises(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    with pytest.raises(ValueError, match="unknown selection strategy"):
        select([x, y], [(0, 1)], strategy="bogus")


def test_select_helper_empty_polynomials_raises():
    with pytest.raises(ValueError, match="polynomial list"):
        select([], [(0, 1)])


def test_select_helper_empty_pairs_raises(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    with pytest.raises(ValueError, match="pair set"):
        select([x, y], [])


# -------- BasicExpert --------


def test_basic_expert_strategy_first_returns_smallest_pair(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = BasicExpert(env, strategy="first")
    assert expert((env.generators, env.pairs)) == env.pairs[0]


def test_basic_expert_strategy_normal_picks_smallest_lcm(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    G = [x**2 + 1, x * y + 1, y**2 + 1]
    P = [(0, 1), (0, 2), (1, 2)]
    env = BuchbergerEnv(DummyIdealGenerator([G]), mode="eval")
    env.reset()
    expert = BasicExpert(env, strategy="normal")
    assert expert((env.generators, env.pairs)) == (1, 2)


def test_basic_expert_strategy_degree(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    G = [x**2 + 1, x * y + 1, y**2 + 1]
    P = [(0, 1), (0, 2), (1, 2)]
    env = BuchbergerEnv(DummyIdealGenerator([G]), mode="eval")
    env.reset()
    expert = BasicExpert(env, strategy="degree")
    assert expert((env.generators, env.pairs)) == (0, 1)


def test_basic_expert_random_strategy_is_seeded(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = BasicExpert(env, strategy="random")
    np.random.seed(0)
    a = expert((env.generators, env.pairs))
    np.random.seed(0)
    b = expert((env.generators, env.pairs))
    assert a == b


# -------- LowestLM / LeastRemainingPairs / ClosestLM --------


def test_lowest_lm_expert_picks_pair_with_min_lm(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = LowestLMExpert(env)
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs


def test_least_remaining_pairs_expert_returns_valid_pair(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = LeastRemainingPairsExpert(env)
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs


def test_closest_lm_expert_returns_valid_pair(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = ClosestLMExpert(env)
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs


def test_closest_lm_expert_update_env_recomputes_basis(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    env = BuchbergerEnv(DummyIdealGenerator([[x**2 + y, x * y + 1]]), mode="eval")
    env.reset()
    expert = ClosestLMExpert(env)
    original_basis = expert.basis

    other_env = BuchbergerEnv(DummyIdealGenerator([[x + y]]), mode="eval")
    other_env.reset()

    expert.update_env(other_env)

    assert expert.env is other_env
    assert expert.basis != original_basis
    assert expert.leading_terms == {p.LM for p in expert.basis}


# -------- helpers --------


def test_next_step_does_not_mutate_original_env(simple_ideal):
    env = _fresh_env(simple_ideal)
    original_gens = list(env.generators)
    original_pairs = list(env.pairs)
    pair = env.pairs[0]

    _, _, new_env = next_step(env, pair)

    assert env.generators == original_gens
    assert env.pairs == original_pairs
    assert new_env is not env


def test_get_basis_returns_groebner_basis(simple_ideal):
    basis = get_basis(simple_ideal)
    assert basis
    assert_reduced_groebner_basis(basis)


def test_get_leading_terms_returns_set_of_monomial_tuples(simple_ideal):
    basis = get_basis(simple_ideal)
    leading_terms = get_leading_terms(basis)
    assert isinstance(leading_terms, set)
    for term in leading_terms:
        assert isinstance(term, tuple)


def test_lm_by_pair_omits_zero_reductions(ring_xy_qq_lex):
    _, x, y = ring_xy_qq_lex
    # spoly(x**2, x*y) reduces to zero.
    G = [x**2, x * y]
    pairs = [(0, 1)]
    env = BuchbergerEnv(DummyIdealGenerator([G]), mode="eval")
    env.reset()
    result = lm_by_pair(env, G, pairs)
    assert (0, 1) not in result


# -------- RolloutExpert / OptimalDPExpert / MCTSExpert --------


def test_rollout_expert_picks_a_valid_pair(simple_ideal):
    env = _fresh_env(simple_ideal)
    base = BasicExpert(env, strategy="first")
    expert = RolloutExpert(env, base_policy=base, n_rollouts=1, gamma=1.0)
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs


def test_optimal_dp_expert_returns_valid_pair_on_small_ideal(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = OptimalDPExpert(env, gamma=1.0, max_states=10_000)
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs


def test_optimal_dp_expert_raises_when_state_budget_exceeded(simple_ideal):
    env = _fresh_env(simple_ideal)
    expert = OptimalDPExpert(env, gamma=1.0, max_states=1)
    with pytest.raises(RuntimeError, match="max_states"):
        expert((env.generators, env.pairs))


def test_mcts_expert_returns_valid_pair_with_low_simulations(simple_ideal):
    env = _fresh_env(simple_ideal)
    rollout = BasicExpert(env, strategy="first")
    np.random.seed(0)
    expert = MCTSExpert(
        env, rollout_policy=rollout, n_simulations=3, c=1.0, gamma=0.99
    )
    pair = expert((env.generators, env.pairs))
    assert pair in env.pairs
