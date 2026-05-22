"""Run Gumbel AlphaZero training on the Buchberger environment."""

import argparse
import os
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

from grobnerRl.env import BuchbergerEnv
from grobnerRl.ideals import RandomBinomialIdealGenerator
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.gumbelMuZero import GumbelAZConfig, train_iteration
from grobnerRl.training.shared import GrainReplayBuffer, TrainConfig

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"


def save_curve(values: list[float], title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(values)), values)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iterations", type=int, default=250)
    parser.add_argument("--num-episodes-per-iter", type=int, default=2)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--max-considered-actions", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs-per-iteration", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=4096)
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--num-generators", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    num_vars: int = args.num_vars

    ideal_gen = RandomBinomialIdealGenerator(
        n=num_vars,
        d=args.max_degree,
        s=args.num_generators,
    )
    env = BuchbergerEnv(ideal_gen)

    config = ModelConfig(
        monomials_dim=num_vars,
        monoms_embedding_dim=32,
        polys_embedding_dim=64,
        poly_embedder_depth=2,
        ideal_depth=2,
        ideal_num_heads=2,
        ideal_hidden_dim=64,
        value_hidden_dim=64,
    )

    key = jax.random.key(args.seed)
    model: GrobnerPolicyValue = GrobnerPolicyValue.from_scratch(config, key)

    optimizer = optax.nadam(args.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    az_cfg = GumbelAZConfig(
        num_simulations=args.num_simulations,
        max_considered_actions=args.max_considered_actions,
        max_episode_steps=args.max_episode_steps,
    )
    train_cfg = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs_per_iteration=args.num_epochs_per_iteration,
    )
    replay = GrainReplayBuffer(max_size=args.replay_size)
    rng = np.random.default_rng(args.seed)

    history: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "mean_reward": [],
        "mean_length": [],
    }

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    for it in range(args.num_iterations):
        model, opt_state, metrics = train_iteration(
            env,
            model,
            optimizer,
            opt_state,
            replay,
            az_cfg,
            train_cfg,
            num_episodes=args.num_episodes_per_iter,
            rng=rng,
            base_seed=args.seed + it * args.num_episodes_per_iter,
        )
        print(f"iter {it}: {metrics}")

        for key_name in history:
            history[key_name].append(float(metrics[key_name]))

    plot_specs: list[tuple[str, str, str, str]] = [
        ("policy_loss", "Policy loss", "loss", "gumbel_az_policy_loss.png"),
        ("value_loss", "Value loss", "loss", "gumbel_az_value_loss.png"),
        ("total_loss", "Total loss", "loss", "gumbel_az_total_loss.png"),
        ("mean_reward", "Mean episode reward", "reward", "gumbel_az_mean_reward.png"),
        ("mean_length", "Mean episode length", "steps", "gumbel_az_mean_length.png"),
    ]
    for metric_key, title, ylabel, filename in plot_specs:
        save_curve(history[metric_key], title, ylabel, filename)
