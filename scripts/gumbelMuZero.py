"""
Gumbel MuZero training script entry point.

This script orchestrates Gumbel MuZero training using consolidated components
from grobnerRl.models, grobnerRl.training.shared, and grobnerRl.training.gumbelMuZero.
"""

import os

import equinox as eqx
import jax
import optax

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.gumbelMuZero import (
    GumbelMuZeroConfig,
    generate_self_play_data,
)
from grobnerRl.training.shared import (
    ReplayBuffer,
    TrainConfig,
    evaluate_model,
    train_policy_value,
)
from grobnerRl.training.utils import save_checkpoint

if __name__ == "__main__":
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)

    pretrained_checkpoint_path: str | None = os.path.join("models", "checkpoints_LML", "best.eqx")

    model_config = ModelConfig(
        monomials_dim=num_vars + 1,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=4,
        ideal_num_heads=8,
        value_hidden_dim=128,
    )

    gumbel_config = GumbelMuZeroConfig(
        num_simulations=25,
        max_num_considered_actions=16,
        gamma=0.99,
        c_visit=50.0,
        c_scale=1.0,
    )

    train_config = TrainConfig(
        learning_rate=1e-4,
        batch_size=128,
        num_epochs_per_iteration=4,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    )

    num_iterations = 50
    episodes_per_iteration = 5
    replay_buffer_size = 2**14
    checkpoint_dir = os.path.join("models", "gumbel_muzero_checkpoints")
    eval_interval = 5
    eval_episodes = 10

    key = jax.random.key(42)

    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_gen, mode="train")

    optimizer = optax.nadam(train_config.learning_rate)

    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained model from {pretrained_checkpoint_path}")
        key, k_model = jax.random.split(key)
        model = GrobnerPolicyValue.from_pretrained(
            checkpoint_path=pretrained_checkpoint_path,
            config=model_config,
            optimizer=optimizer,
            key=k_model,
        )
        print("Pretrained policy loaded. Fresh value head initialized.")
    else:
        print("Initializing model from scratch")
        key, k_model = jax.random.split(key)
        model = GrobnerPolicyValue.from_scratch(config=model_config, key=k_model)

    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    print("\nStarting Gumbel MuZero training...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Episodes per iteration: {episodes_per_iteration}")
    print(f"  Simulations: {gumbel_config.num_simulations}")
    print(f"  Max considered actions: {gumbel_config.max_num_considered_actions}")
    print(f"  Replay buffer size: {replay_buffer_size}")
    print(f"  Checkpoint directory: {checkpoint_dir}")

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    best_reward = float("-inf")

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        print("\nGenerating self-play data...")
        key, subkey = jax.random.split(key)
        experiences = generate_self_play_data(
            model, env, episodes_per_iteration, gumbel_config, subkey
        )
        print(f"Generated {len(experiences)} experiences")

        replay_buffer.add(experiences)
        print(f"Replay buffer size: {len(replay_buffer)}")

        metrics: dict = {}
        if len(replay_buffer) >= train_config.batch_size:
            print("\nTraining...")
            model, opt_state, metrics = train_policy_value(
                model, replay_buffer, train_config, optimizer, opt_state
            )
            print(
                f"  Policy loss: {metrics['policy_loss']:.4f}, "
                f"Value loss: {metrics['value_loss']:.4f}, "
                f"Total loss: {metrics['total_loss']:.4f}"
            )

            if checkpoint_dir:
                save_checkpoint(
                    model, opt_state, checkpoint_dir, "last", iteration + 1, metrics
                )

        if (iteration + 1) % eval_interval == 0:
            print("\nEvaluating...")
            eval_metrics = evaluate_model(model, env, eval_episodes)
            print(
                f"  Mean reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}, "
                f"Mean length: {eval_metrics['mean_length']:.1f}"
            )

            if eval_metrics["mean_reward"] > best_reward:
                best_reward = eval_metrics["mean_reward"]
                if checkpoint_dir:
                    combined_metrics = {**metrics, **eval_metrics}
                    save_checkpoint(
                        model, opt_state, checkpoint_dir, "best_gumbel_muzero", iteration + 1, combined_metrics
                    )
                    print(f"  Saved new best model (reward: {best_reward:.2f})")

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print("\nDone!")
