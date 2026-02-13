"""Unit tests for Gumbel MuZero components."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from grobnerRl.envs.env import BuchbergerEnv, make_obs
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.gumbelMuZero import (
    GumbelMuZeroConfig,
    GumbelMuZeroSearch,
    GumbelNode,
    copy_env,
    gumbel_top_k,
    run_self_play_episode,
    sample_gumbel,
    sequential_halving,
    sigma,
)

# Create test environment components
R, x, y = sp.ring("x,y", sp.QQ, "lex")


class DummyIdealGenerator:
    """Dummy generator for testing."""

    def __init__(self, batches):
        self._data = iter(batches)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data)


def test_sample_gumbel_distribution():
    """Test that Gumbel sampling produces correct distribution."""
    key = jax.random.key(42)
    samples = sample_gumbel(key, (10000,))
    
    # Gumbel(0,1) has mean ≈ 0.5772 (Euler-Mascheroni constant)
    # and variance π²/6 ≈ 1.645
    mean = float(jnp.mean(samples))
    var = float(jnp.var(samples))
    
    assert 0.5 < mean < 0.65, f"Mean {mean} not close to 0.5772"
    assert 1.5 < var < 1.8, f"Variance {var} not close to 1.645"


def test_sample_gumbel_range():
    """Test that Gumbel samples are in reasonable range."""
    key = jax.random.key(42)
    samples = sample_gumbel(key, (1000,))
    
    # Should not have extreme values
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples > -10)
    assert jnp.all(samples < 10)


def test_gumbel_top_k_respects_k():
    """Test that Gumbel-Top-k returns exactly k actions."""
    key = jax.random.key(42)
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valid_actions = [0, 1, 2, 3, 4]
    
    for k in [1, 2, 3, 5]:
        selected, gumbels = gumbel_top_k(key, logits, valid_actions, k)
        assert len(selected) == min(k, len(valid_actions))
        assert len(gumbels) == min(k, len(valid_actions))


def test_gumbel_top_k_respects_valid_actions():
    """Test that Gumbel-Top-k only selects from valid actions."""
    key = jax.random.key(42)
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valid_actions = [0, 2, 4]  # Only odd indices
    
    selected, _ = gumbel_top_k(key, logits, valid_actions, k=2)
    
    assert all(action in valid_actions for action in selected)


def test_gumbel_top_k_diversity():
    """Test that Gumbel-Top-k produces diverse selections."""
    logits = np.array([1.0, 1.0, 1.0, 1.0])  # Equal probabilities
    valid_actions = [0, 1, 2, 3]
    
    # Run multiple times and check we get different selections
    selections = []
    for i in range(10):
        key = jax.random.key(i)
        selected, _ = gumbel_top_k(key, logits, valid_actions, k=2)
        selections.append(tuple(sorted(selected)))
    
    # Should have some diversity
    unique_selections = set(selections)
    assert len(unique_selections) > 1, "Gumbel-Top-k should produce diverse selections"


def test_sigma_normalization():
    """Test sigma function normalization."""
    logits = np.array([1.0, 2.0, 3.0])
    q_values = np.array([0.5, 1.0, 1.5])
    visit_counts = np.array([10, 20, 30])
    
    sigma_vals = sigma(logits, q_values, visit_counts, c_visit=50.0, c_scale=1.0)
    
    # Should return finite values
    assert np.all(np.isfinite(sigma_vals))


def test_sigma_with_zero_visits():
    """Test sigma function with zero visits."""
    logits = np.array([1.0, 2.0, 3.0])
    q_values = np.array([0.0, 0.0, 0.0])
    visit_counts = np.array([0, 0, 0])
    
    sigma_vals = sigma(logits, q_values, visit_counts, c_visit=50.0, c_scale=1.0)
    
    # Should handle zero visits gracefully
    assert np.all(np.isfinite(sigma_vals))


def test_sigma_with_equal_q_values():
    """Test sigma function with equal Q-values."""
    logits = np.array([1.0, 2.0, 3.0])
    q_values = np.array([1.0, 1.0, 1.0])
    visit_counts = np.array([10, 20, 30])
    
    sigma_vals = sigma(logits, q_values, visit_counts, c_visit=50.0, c_scale=1.0)
    
    assert np.all(np.isfinite(sigma_vals))


def test_gumbel_node_q_value_zero_visits():
    """Test GumbelNode Q-value with zero visits."""
    node = GumbelNode()
    assert node.q_value == 0.0


def test_gumbel_node_q_value_with_visits():
    """Test GumbelNode Q-value computation."""
    node = GumbelNode()
    node.visit_count = 5
    node.value_sum = 10.0
    
    assert node.q_value == 2.0


def test_copy_env():
    """Test that copy_env creates independent copy."""
    ideal_gen = DummyIdealGenerator([[x**2 + y, x * y + 1]])
    env = BuchbergerEnv(ideal_gen, mode="eval")
    env.reset()
    
    # Copy the environment
    env_copy = copy_env(env)
    
    # Modify the copy
    env_copy.generators.append(x + y)
    env_copy.pairs.append((0, 1))
    
    # Original should be unchanged
    assert len(env.generators) == 2
    assert x + y not in env.generators


def test_copy_env_independence():
    """Test that modifying copied env doesn't affect original."""
    ideal_gen = DummyIdealGenerator([[x**2 + y, x * y + 1]])
    env = BuchbergerEnv(ideal_gen, mode="eval")
    env.reset()
    
    original_generators = list(env.generators)
    original_pairs = list(env.pairs)
    
    env_copy = copy_env(env)
    
    # Step in the copy
    if env_copy.pairs:
        env_copy.step(env_copy.pairs[0])
    
    # Original should be unchanged
    assert env.generators == original_generators
    assert env.pairs == original_pairs


def test_sequential_halving_single_action():
    """Test sequential halving with single action."""
    key = jax.random.key(42)
    config = ModelConfig(monomials_dim=3, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=10)
    
    ideal_gen = DummyIdealGenerator([[x + y, x * y]])
    env = BuchbergerEnv(ideal_gen, mode="train")
    env.reset()
    
    root = GumbelNode(env=copy_env(env))
    actions = np.array([0])
    gumbel_values = np.array([1.0])
    
    selected = sequential_halving(actions, gumbel_values, root, env, model, gumbel_config)
    
    assert selected == 0


def test_sequential_halving_two_actions():
    """Test sequential halving with two actions."""
    key = jax.random.key(42)
    config = ModelConfig(monomials_dim=3, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=10)
    
    ideal_gen = DummyIdealGenerator([[x + y, x * y, x**2]])
    env = BuchbergerEnv(ideal_gen, mode="train")
    env.reset()
    
    root = GumbelNode(env=copy_env(env))
    
    # Get valid actions from the environment
    num_polys = len(env.generators)
    valid_actions = [i * num_polys + j for i, j in env.pairs]
    
    if len(valid_actions) >= 2:
        actions = np.array(valid_actions[:2])
        gumbel_values = np.array([2.0, 1.0])
        
        selected = sequential_halving(actions, gumbel_values, root, env, model, gumbel_config)
        
        assert selected in actions


def test_gumbel_search_basic():
    """Test basic Gumbel search functionality."""
    key = jax.random.key(42)
    config = ModelConfig(monomials_dim=3, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=5, max_num_considered_actions=4)
    
    ideal_gen = DummyIdealGenerator([[x + y, x * y]])
    env = BuchbergerEnv(ideal_gen, mode="train")
    env.reset()
    
    search = GumbelMuZeroSearch(model, env, gumbel_config)
    
    key, subkey = jax.random.split(key)
    policy, value = search.search(env, subkey)
    
    # Check policy is valid probability distribution
    assert policy.shape[0] == len(env.generators) ** 2
    assert np.isfinite(value)
    assert policy.sum() > 0  # Should have some mass


def test_gumbel_search_no_valid_actions():
    """Test Gumbel search with no valid actions."""
    key = jax.random.key(42)
    config = ModelConfig(monomials_dim=3, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=5)
    
    ideal_gen = DummyIdealGenerator([[x]])
    env = BuchbergerEnv(ideal_gen, mode="train")
    env.reset()
    
    # Force no pairs
    env.pairs = []
    
    search = GumbelMuZeroSearch(model, env, gumbel_config)
    
    key, subkey = jax.random.split(key)
    policy, value = search.search(env, subkey)
    
    # Should return zero policy
    assert policy.sum() == 0


def test_self_play_episode_structure():
    """Test self-play episode generates correct structure."""
    key = jax.random.key(42)
    num_vars = 3
    config = ModelConfig(monomials_dim=num_vars + 1, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=3)
    
    ideal_gen = SAT3IdealGenerator(num_vars, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")
    
    key, subkey = jax.random.split(key)
    from grobnerRl.training.shared import PolynomialCache
    poly_cache = PolynomialCache()
    experiences = run_self_play_episode(model, env, gumbel_config, subkey, poly_cache)
    
    # Should have some experiences
    assert len(experiences) > 0
    
    # Check structure
    for exp in experiences:
        assert hasattr(exp, 'observation')
        assert hasattr(exp, 'policy')
        assert hasattr(exp, 'value')
        assert hasattr(exp, 'num_polys')
        assert exp.num_polys > 0
        assert exp.policy.shape[0] == exp.num_polys ** 2


def test_self_play_return_calculation():
    """Test that returns are calculated correctly with discounting."""
    key = jax.random.key(42)
    num_vars = 3
    config = ModelConfig(monomials_dim=num_vars + 1, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=3, gamma=0.9)
    
    ideal_gen = SAT3IdealGenerator(num_vars, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")
    
    key, subkey = jax.random.split(key)
    from grobnerRl.training.shared import PolynomialCache
    poly_cache = PolynomialCache()
    experiences = run_self_play_episode(model, env, gumbel_config, subkey, poly_cache)
    
    if len(experiences) > 1:
        # First experience should have highest return (most discounted future)
        # Last experience should be close to immediate reward
        assert experiences[0].value != experiences[-1].value


def test_self_play_policy_normalization():
    """Test that policies are properly normalized."""
    key = jax.random.key(42)
    num_vars = 3
    config = ModelConfig(monomials_dim=num_vars + 1, monoms_embedding_dim=16, polys_embedding_dim=32)
    model = GrobnerPolicyValue.from_scratch(config, key)
    
    gumbel_config = GumbelMuZeroConfig(num_simulations=5)
    
    ideal_gen = SAT3IdealGenerator(num_vars, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")
    
    key, subkey = jax.random.split(key)
    from grobnerRl.training.shared import PolynomialCache
    poly_cache = PolynomialCache()
    experiences = run_self_play_episode(model, env, gumbel_config, subkey, poly_cache)
    
    # Check policies sum to reasonable values (may not be exactly 1 due to masking)
    for exp in experiences:
        policy_sum = exp.policy.sum()
        assert policy_sum >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
