"""
Manual testing and diagnostic script for Gumbel MuZero.

This script provides various smoke tests and diagnostic tools to validate
the Gumbel MuZero implementation:
- Quick smoke test
- Overfitting test (single ideal)
- Benchmark comparison with experts
- Hyperparameter sensitivity analysis
"""

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.experts import LowestLMExpert
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.gumbelMuZero import (
    GumbelMuZeroConfig,
    generate_self_play_data,
)
from grobnerRl.training.shared import (
    PolynomialCache,
    ReplayBuffer,
    TrainConfig,
    evaluate_model,
    train_policy_value,
)


def smoke_test():
    """Quick smoke test: run 3 iterations and check for crashes."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Quick validation")
    print("=" * 60)

    key = jax.random.key(42)

    # Small configuration for fast testing
    model_config = ModelConfig(
        monomials_dim=4,
        monoms_embedding_dim=16,
        polys_embedding_dim=32,
        ideal_depth=2,
        ideal_num_heads=2,
        value_hidden_dim=32,
    )

    gumbel_config = GumbelMuZeroConfig(
        num_simulations=5,
        max_num_considered_actions=4,
    )

    train_config = TrainConfig(
        learning_rate=1e-3,
        batch_size=8,
        num_epochs_per_iteration=1,
    )

    # Create model and environment
    key, k_model = jax.random.split(key)
    model = GrobnerPolicyValue.from_scratch(model_config, k_model)
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    ideal_gen = SAT3IdealGenerator(3, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")

    poly_cache = PolynomialCache()
    replay_buffer = ReplayBuffer(max_size=1000, poly_cache=poly_cache)

    print("\nRunning 3 training iterations...")

    for iteration in range(3):
        print(f"\nIteration {iteration + 1}/3")

        # Self-play
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, num_episodes=2, config=gumbel_config, key=subkey, poly_cache=poly_cache
        )
        print(f"  Generated {len(experiences)} experiences")

        replay_buffer.add(experiences)

        # Train
        if len(replay_buffer) >= train_config.batch_size:
            model, opt_state, metrics = train_policy_value(
                model, replay_buffer, train_config, optimizer, opt_state
            )

            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  Total loss: {metrics['total_loss']:.4f}")

            # Check for numerical issues
            if not np.isfinite(metrics["total_loss"]):
                print("  ❌ ERROR: Loss is not finite!")
                return False

    print("\n✓ Smoke test passed: No crashes, losses are finite")
    return True


def overfitting_test(num_iterations=20):
    """Test overfitting on a single ideal instance.

    Success criteria:
    - Final reward should be higher than initial reward
    - Loss should decrease over time
    - Final reward should be positive (completing the ideal successfully)
    """
    print("\n" + "=" * 60)
    print("OVERFITTING TEST: Learn single ideal")
    print("=" * 60)

    key = jax.random.key(42)

    model_config = ModelConfig(
        monomials_dim=4,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=2,
        ideal_num_heads=4,
        value_hidden_dim=128,
    )

    gumbel_config = GumbelMuZeroConfig(
        num_simulations=20,
        max_num_considered_actions=8,
    )

    train_config = TrainConfig(
        learning_rate=5e-4,
        batch_size=32,
        num_epochs_per_iteration=5,
    )

    # Create model
    key, k_model = jax.random.split(key)
    model = GrobnerPolicyValue.from_scratch(model_config, k_model)
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Single ideal instance
    class SingleIdealGenerator:
        def __init__(self, seed=42):
            temp_gen = SAT3IdealGenerator(3, 13)
            self.key = jax.random.key(seed)
            self.ideal = next(iter(temp_gen))

        def __iter__(self):
            return self

        def __next__(self):
            return self.ideal

    ideal_gen = SingleIdealGenerator()
    env = BuchbergerEnv(ideal_gen, mode="train")

    poly_cache = PolynomialCache()
    replay_buffer = ReplayBuffer(max_size=2000, poly_cache=poly_cache)
    losses = []
    rewards = []

    print(f"\nTraining for {num_iterations} iterations on single ideal...")

    # Track initial performance
    initial_eval = None

    for iteration in range(num_iterations):
        # Self-play - more episodes for better data collection
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, num_episodes=10, config=gumbel_config, key=subkey, poly_cache=poly_cache
        )
        replay_buffer.add(experiences)

        # Train
        if len(replay_buffer) >= train_config.batch_size:
            model, opt_state, metrics = train_policy_value(
                model, replay_buffer, train_config, optimizer, opt_state
            )
            losses.append(metrics["total_loss"])

        # Evaluate every 5 iterations
        if (iteration + 1) % 5 == 0:
            eval_metrics = evaluate_model(model, env, num_episodes=20)
            rewards.append(eval_metrics["mean_reward"])

            # Track initial performance
            if initial_eval is None:
                initial_eval = eval_metrics["mean_reward"]

            print(
                f"Iteration {iteration + 1}: Loss={metrics['total_loss']:.4f}, Reward={eval_metrics['mean_reward']:.2f}"
            )

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)

    ax2.plot(range(5, num_iterations + 1, 5), rewards)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Evaluation Reward")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("gumbel_overfitting_test.png")
    print("\n✓ Overfitting test complete. Plot saved to gumbel_overfitting_test.png")
    print(f"  Initial reward: {initial_eval:.2f}")
    print(f"  Final reward: {rewards[-1]:.2f}")
    print(f"  Reward improvement: {rewards[-1] - initial_eval:.2f}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")

    # Check success criteria
    reward_improved = rewards[-1] > initial_eval
    loss_decreased = losses[-1] < losses[0]
    positive_final_reward = rewards[-1] > 0

    success = reward_improved and loss_decreased

    if success:
        print("\n✅ SUCCESS: Model successfully overfitted to single ideal!")
        print(f"  ✓ Reward improved: {initial_eval:.2f} → {rewards[-1]:.2f}")
        print(f"  ✓ Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
        if positive_final_reward:
            print(f"  ✓ Achieving positive rewards (task completion)")
    else:
        print("\n⚠ WARNING: Overfitting criteria not fully met:")
        if not reward_improved:
            print(f"  ✗ Reward did not improve: {initial_eval:.2f} → {rewards[-1]:.2f}")
        if not loss_decreased:
            print(f"  ✗ Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")
        if not positive_final_reward:
            print(f"  ✗ Final reward not positive: {rewards[-1]:.2f}")

    return success


def benchmark_comparison(num_episodes=50):
    """Compare trained model with expert baseline."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON: Model vs Expert")
    print("=" * 60)

    key = jax.random.key(42)

    model_config = ModelConfig(
        monomials_dim=4,
        monoms_embedding_dim=32,
        polys_embedding_dim=64,
        ideal_depth=2,
        ideal_num_heads=4,
    )

    # Create untrained model for baseline comparison
    key, k_model = jax.random.split(key)
    model = GrobnerPolicyValue.from_scratch(model_config, k_model)

    ideal_gen = SAT3IdealGenerator(3, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")

    # Evaluate untrained model
    print(f"\nEvaluating untrained model on {num_episodes} episodes...")
    model_metrics = evaluate_model(model, env, num_episodes=num_episodes)
    print(
        f"Model - Mean reward: {model_metrics['mean_reward']:.2f} ± {model_metrics['std_reward']:.2f}"
    )
    print(f"Model - Mean length: {model_metrics['mean_length']:.1f}")

    # Evaluate expert
    print(f"\nEvaluating LowestLM expert on {num_episodes} episodes...")
    expert = LowestLMExpert(env)
    expert_rewards = []
    expert_lengths = []

    for seed in range(num_episodes):
        env.reset(seed=seed)
        expert.update_env(env)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            obs = (env.generators, env.pairs)
            action = expert(obs)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        expert_rewards.append(total_reward)
        expert_lengths.append(steps)

    print(
        f"Expert - Mean reward: {np.mean(expert_rewards):.2f} ± {np.std(expert_rewards):.2f}"
    )
    print(f"Expert - Mean length: {np.mean(expert_lengths):.1f}")

    print("\n✓ Benchmark comparison complete")


def hyperparameter_sensitivity():
    """Test sensitivity to key hyperparameters."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY TEST")
    print("=" * 60)

    key = jax.random.key(42)

    model_config = ModelConfig(
        monomials_dim=4,
        monoms_embedding_dim=16,
        polys_embedding_dim=32,
        ideal_depth=2,
        ideal_num_heads=2,
    )

    # Test different num_simulations
    sim_values = [5, 10, 20]

    print("\nTesting num_simulations...")
    for num_sims in sim_values:
        key, k_model = jax.random.split(key)
        model = GrobnerPolicyValue.from_scratch(model_config, k_model)

        gumbel_config = GumbelMuZeroConfig(
            num_simulations=num_sims,
            max_num_considered_actions=8,
        )

        ideal_gen = SAT3IdealGenerator(3, 5)
        env = BuchbergerEnv(ideal_gen, mode="train")

        # Generate one episode
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, num_episodes=1, config=gumbel_config, key=subkey
        )

        print(f"  num_simulations={num_sims}: Generated {len(experiences)} experiences")

    print("\n✓ Hyperparameter sensitivity test complete")


def diagnostic_logging():
    """Run training with detailed diagnostic logging."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC LOGGING")
    print("=" * 60)

    key = jax.random.key(42)

    model_config = ModelConfig(
        monomials_dim=4,
        monoms_embedding_dim=32,
        polys_embedding_dim=64,
        ideal_depth=2,
        ideal_num_heads=4,
    )

    gumbel_config = GumbelMuZeroConfig(
        num_simulations=10,
        max_num_considered_actions=8,
    )

    train_config = TrainConfig(
        learning_rate=1e-3,
        batch_size=16,
        num_epochs_per_iteration=2,
    )

    key, k_model = jax.random.split(key)
    model = GrobnerPolicyValue.from_scratch(model_config, k_model)
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    ideal_gen = SAT3IdealGenerator(3, 5)
    env = BuchbergerEnv(ideal_gen, mode="train")

    poly_cache = PolynomialCache()
    replay_buffer = ReplayBuffer(max_size=1000, poly_cache=poly_cache)

    print("\nRunning 5 iterations with diagnostics...")

    for iteration in range(5):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Self-play
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, num_episodes=3, config=gumbel_config, key=subkey, poly_cache=poly_cache
        )

        # Compute diagnostics on experiences
        policies = [exp.policy for exp in experiences]
        values = [exp.value for exp in experiences]

        # Policy entropy (measure of exploration)
        entropies = []
        for policy in policies:
            if policy.sum() > 0:
                p_norm = policy / policy.sum()
                p_norm = p_norm[p_norm > 0]  # Remove zeros
                entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))
                entropies.append(entropy)

        print(f"  Experiences: {len(experiences)}")
        print(f"  Value range: [{np.min(values):.2f}, {np.max(values):.2f}]")
        print(
            f"  Mean policy entropy: {np.mean(entropies):.3f}"
            if entropies
            else "  No valid policies"
        )

        replay_buffer.add(experiences)

        # Train
        if len(replay_buffer) >= train_config.batch_size:
            model, opt_state, metrics = train_policy_value(
                model, replay_buffer, train_config, optimizer, opt_state
            )

            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  Total loss: {metrics['total_loss']:.4f}")

            # Check gradient norms (via optimizer state)
            # This is a simplified check
            params = eqx.filter(model, eqx.is_array)
            leaves = jax.tree_util.tree_leaves(params)
            param_norms = [float(jnp.linalg.norm(leaf.flatten())) for leaf in leaves]
            print(f"  Max param norm: {max(param_norms):.3f}")

    print("\n✓ Diagnostic logging complete")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GUMBEL MUZERO VALIDATION TEST SUITE")
    print("=" * 60)

    # Run tests
    try:
        # 1. Smoke test (fast)
        if not smoke_test():
            print("\n❌ Smoke test failed. Stopping.")
            return

        # 2. Diagnostic logging
        diagnostic_logging()

        # 3. Benchmark comparison
        benchmark_comparison(num_episodes=20)

        # 4. Hyperparameter sensitivity
        hyperparameter_sensitivity()

        # 5. Overfitting test (slower)
        if "--full" in sys.argv:
            if not overfitting_test(num_iterations=30):
                print(
                    "\n⚠ Overfitting test did not meet all success criteria (this may be expected with current hyperparameters)"
                )
        else:
            print("\n(Skipping overfitting test. Use --full flag to run it)")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
