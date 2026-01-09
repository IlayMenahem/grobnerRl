"""
AlphaZero-style MCTS training for Buchberger environment.

This script implements:
- MCTS with neural network guidance (policy + value)
- Self-play data generation
- Combined policy and value training
- Support for pretrained models from supervised_jax.py
"""

import os

import jax
import optax

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.training.alphaZero import ModelConfig, MCTSConfig, TrainConfig, GrobnerAlphaZero, ReplayBuffer, alphazero_training_loop


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Environment configuration
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)

    # Pretrained model (set to None to train from scratch)
    pretrained_checkpoint_path: str | None = "models/checkpoints/best.eqx"

    # Model configuration
    model_config = ModelConfig(
        monomials_dim=num_vars + 1,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=4,
        ideal_num_heads=8,
        value_hidden_dim=128,
    )

    # MCTS configuration
    mcts_config = MCTSConfig(
        num_simulations=25,  # Reduced for faster iteration
        c_puct=1.0,
        gamma=0.99,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    # Training configuration
    train_config = TrainConfig(
        learning_rate=1e-4,
        batch_size=32,
        num_epochs_per_iteration=3,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    )

    # AlphaZero loop configuration
    num_iterations = 50
    episodes_per_iteration = 5
    replay_buffer_size = 50000
    checkpoint_dir = os.path.join("models", "alphazero_checkpoints")
    eval_interval = 5
    eval_episodes = 10

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Random key
    key = jax.random.key(42)

    # Create environment
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_gen, mode="train")

    # Create optimizer
    optimizer = optax.nadam(train_config.learning_rate)

    # Initialize model
    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained model from {pretrained_checkpoint_path}")
        key, k_model = jax.random.split(key)
        model = GrobnerAlphaZero.from_pretrained(
            checkpoint_path=pretrained_checkpoint_path,
            config=model_config,
            optimizer=optimizer,
            key=k_model,
        )
        print("Pretrained policy loaded. Fresh value head initialized.")
    else:
        print("Initializing model from scratch")
        key, k_model = jax.random.split(key)
        model = GrobnerAlphaZero.from_scratch(
            config=model_config,
            key=k_model,
        )

    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("\nStarting AlphaZero training...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Episodes per iteration: {episodes_per_iteration}")
    print(f"  MCTS simulations: {mcts_config.num_simulations}")
    print(f"  Replay buffer size: {replay_buffer_size}")
    print(f"  Checkpoint directory: {checkpoint_dir}")

    trained_model = alphazero_training_loop(
        model=model,
        env=env,
        num_iterations=num_iterations,
        episodes_per_iteration=episodes_per_iteration,
        mcts_config=mcts_config,
        train_config=train_config,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        checkpoint_dir=checkpoint_dir,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    print("\nDone!")

