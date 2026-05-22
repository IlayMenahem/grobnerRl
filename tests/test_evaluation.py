"""Tests for grobnerRl.evaluation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from matplotlib.figure import Figure

from grobnerRl.env import BuchbergerEnv
from grobnerRl.evaluation import (
    _greedy_pair,
    benchmark,
    evaluate_agent,
    evaluate_expert,
    evaluate_model,
    make_model_agent,
    plot_episode_rewards,
    run_episode,
)
from grobnerRl.experts import BasicExpert
from grobnerRl.models import GrobnerPolicyValue, ModelConfig

from tests.conftest import DummyIdealGenerator


def _model_for(num_vars: int) -> GrobnerPolicyValue:
    return GrobnerPolicyValue.from_scratch(
        ModelConfig(monomials_dim=num_vars),
        jax.random.PRNGKey(0),
    )


def test_run_episode_returns_sum_of_rewards(simple_ideal):
    env = BuchbergerEnv(
        DummyIdealGenerator([simple_ideal]),
        mode="eval",
        rewards="reductions",
    )
    expert = BasicExpert(env, strategy="first")
    expert.update_env(env)

    total = run_episode(env, expert)

    # The simple ideal terminates after a small number of steps; each non-terminal
    # step contributes -1 under the "reductions" reward scheme.
    assert total <= 0
    assert isinstance(total, float)


def test_run_episode_forwards_seed_to_reset():
    fake_env = MagicMock()
    fake_env.reset.return_value = (None, {})
    fake_env.step.return_value = (None, -1.0, True, False, {})
    agent = MagicMock(return_value=(0, 1))

    run_episode(fake_env, agent, seed=42)

    fake_env.reset.assert_called_once_with(seed=42)


def test_evaluate_agent_uses_incremental_seeds():
    fake_env = MagicMock()
    fake_env.reset.return_value = (None, {})
    fake_env.step.return_value = (None, -1.0, True, False, {})
    agent = MagicMock(return_value=(0, 1))

    evaluate_agent(agent, fake_env, num_episodes=3, base_seed=10)

    seeds = [call.kwargs.get("seed") for call in fake_env.reset.call_args_list]
    assert seeds == [10, 11, 12]


def test_evaluate_agent_returns_float64_array_of_correct_length(simple_ideal):
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal, simple_ideal]), mode="eval")
    expert = BasicExpert(env, strategy="first")
    expert.update_env(env)

    rewards = evaluate_agent(expert, env, num_episodes=2)

    assert rewards.shape == (2,)
    assert rewards.dtype == np.float64


def test_evaluate_expert_updates_env_before_running(simple_ideal):
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal]), mode="eval")
    expert = BasicExpert(env, strategy="first")
    expert.update_env = MagicMock(wraps=expert.update_env)  # type: ignore[method-assign]

    evaluate_expert(expert, env, num_episodes=1)

    expert.update_env.assert_called_once_with(env)


def test_greedy_pair_picks_max_valid_pair():
    logits = np.array([0.0, 1.0, 2.0, 3.0])
    pairs = [(0, 1), (1, 0)]  # → flat indices 1 and 2 valid
    assert _greedy_pair(logits, pairs, num_polys=2) == (1, 0)  # idx 2 wins


def test_greedy_pair_ignores_invalid_pair_indices():
    logits = np.array([0.0, 10.0, 1.0, 0.0])  # idx 1 invalid, idx 2 valid
    pairs = [(1, 0)]  # only (1,0) valid → flat idx 2
    assert _greedy_pair(logits, pairs, num_polys=2) == (1, 0)


def test_make_model_agent_returns_pair_in_current_env_pairs(simple_ideal):
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal]), mode="eval")
    env.reset()
    model = _model_for(num_vars=2)
    agent = make_model_agent(model, env)

    action = agent(None)
    assert action in env.pairs


def test_evaluate_model_returns_array_of_correct_length(ring_xy_qq_lex):
    # Use an ideal whose only S-pair reduces to zero so the episode terminates
    # after one step without growing the generator count past the model's
    # padded size (a known mismatch in _greedy_pair when n_polys is not a
    # power of two).
    _, x, y = ring_xy_qq_lex
    ideal = [x**2, x * y]
    env = BuchbergerEnv(DummyIdealGenerator([ideal, ideal]), mode="eval")
    model = _model_for(num_vars=2)

    rewards = evaluate_model(model, env, num_episodes=2)

    assert rewards.shape == (2,)
    assert rewards.dtype == np.float64


def test_plot_episode_rewards_writes_png_and_returns_figure(tmp_path: Path):
    rewards = {"a": np.array([-1.0, -2.0]), "b": np.array([-3.0, -1.0])}
    fig = plot_episode_rewards(rewards, filename="out.png", figs_dir=tmp_path)
    assert isinstance(fig, Figure)
    assert (tmp_path / "out.png").exists()


def test_plot_episode_rewards_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one agent"):
        plot_episode_rewards({}, figs_dir=tmp_path)


def test_benchmark_runs_each_agent_and_writes_plot(simple_ideal, tmp_path: Path):
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal] * 4), mode="eval")

    def agent_first(_obs):
        return env.pairs[0]

    def agent_last(_obs):
        return env.pairs[-1]

    results, fig = benchmark(
        {"first": agent_first, "last": agent_last},
        env,
        num_episodes=2,
        filename="bench.png",
        figs_dir=tmp_path,
    )

    assert set(results.keys()) == {"first", "last"}
    assert results["first"].shape == (2,)
    assert results["last"].shape == (2,)
    assert isinstance(fig, Figure)
    assert (tmp_path / "bench.png").exists()
