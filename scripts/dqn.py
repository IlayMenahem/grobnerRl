"""
Rainbow DQN training script entry point.

Mirrors the structure of supervised.py and gumbelMuZero.py:
creates the environment, model, and optimizer, then delegates
all training to grobnerRl.training.dqn.train_rainbow_dqn.
"""

import os

import jax
import optax

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.dqn import RainbowConfig, train_rainbow_dqn

if __name__ == "__main__":
    # --- Problem specification ---
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)

    pretrained_checkpoint_path: str | None = None
    # pretrained_checkpoint_path = os.path.join("models", "checkpoints", "best.eqx")

    # --- Model architecture ---
    model_config = ModelConfig(
        monomials_dim=num_vars + 1,
        monoms_embedding_dim=64,
        polys_embedding_dim=128,
        ideal_depth=2,
        ideal_num_heads=2,
        value_hidden_dim=128,
    )

    # --- Rainbow hyper-parameters ---
    rainbow_config = RainbowConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        target_update_interval=200,
        tau=1.0,
        replay_buffer_size=2**14,
        min_replay_size=512,
        n_step=1,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_epsilon=1e-6,
        num_iterations=10_000,
        train_steps_per_iteration=1,
        eval_interval=100,
        eval_episodes=25,
        checkpoint_dir=os.path.join("models", "rainbow_dqn_checkpoints"),
        logs_dir="logs",
    )

    # --- Initialise model ---
    key = jax.random.key(42)
    optimizer = optax.nadam(rainbow_config.learning_rate)

    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    # Eval mode: env returns raw symbolic observations required by the Rainbow
    # loop for n-step buffering and valid-action masking.
    env = BuchbergerEnv(ideal_gen, mode="eval")

    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained model from {pretrained_checkpoint_path}")
        key, k_model = jax.random.split(key)
        model = GrobnerPolicyValue.from_pretrained(
            checkpoint_path=pretrained_checkpoint_path,
            config=model_config,
            optimizer=optimizer,
            key=k_model,
        )
        print("Pretrained policy loaded.")
    else:
        print("Initialising model from scratch.")
        key, k_model = jax.random.split(key)
        model = GrobnerPolicyValue.from_scratch(config=model_config, key=k_model)

    # --- Print configuration summary ---
    print("\nStarting Rainbow DQN training...")
    print(f"  Problem      : {num_vars}-var SAT3, {num_clauses} clauses")
    print(f"  Iterations   : {rainbow_config.num_iterations}")
    print(f"  Batch size   : {rainbow_config.batch_size}")
    print(f"  n-step       : {rainbow_config.n_step}")
    print(f"  PER alpha    : {rainbow_config.per_alpha}")
    print(f"  Replay size  : {rainbow_config.replay_buffer_size}")
    print(f"  Target sync  : every {rainbow_config.target_update_interval} steps")
    print(f"  Checkpoints  : {rainbow_config.checkpoint_dir}")

    # --- Run training ---
    trained_model, history = train_rainbow_dqn(model, env, optimizer, rainbow_config)
