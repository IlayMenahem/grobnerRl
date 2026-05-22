"""Benchmarking utilities for models and experts.

Provides episode rollouts and per-episode reward plots with mean lines so that
heuristic experts and trained models can be compared on the same problem set.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from grobnerRl.env import BaseEnv, BuchbergerEnv, make_obs
from grobnerRl.experts import Expert
from grobnerRl.models import GrobnerPolicyValue

Action = int | tuple[int, int]
Agent = Callable[[Any], Action]

FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"


def run_episode(env: BaseEnv, agent: Agent, seed: int | None = None) -> float:
    """Roll out one episode using ``agent`` and return the total reward."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = agent(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
    return total_reward


def evaluate_agent(
    agent: Agent,
    env: BaseEnv,
    num_episodes: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Run an agent for ``num_episodes`` and return per-episode rewards."""
    rewards = np.empty(num_episodes, dtype=np.float64)
    for i in range(num_episodes):
        rewards[i] = run_episode(env, agent, seed=base_seed + i)
    return rewards


def evaluate_expert(
    expert: Expert,
    env: BaseEnv,
    num_episodes: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Benchmark an :class:`Expert` over ``num_episodes`` episodes."""
    expert.update_env(env)
    return evaluate_agent(expert, env, num_episodes, base_seed)


def _greedy_pair(
    logits: np.ndarray,
    pairs: list[tuple[int, int]],
    num_polys: int,
) -> tuple[int, int]:
    """Pick the highest-scoring valid pair from a flat policy logits vector."""
    mask = np.full(logits.shape, -np.inf, dtype=np.float64)
    for i, j in pairs:
        mask[i * num_polys + j] = 0.0
    action = int(np.argmax(logits + mask))
    return action // num_polys, action % num_polys


def make_model_agent(model: GrobnerPolicyValue, env: BuchbergerEnv) -> Agent:
    """Wrap ``model`` so it can act as an :data:`Agent` on a ``BuchbergerEnv``.

    The wrapper tokenizes the current env state with :func:`make_obs`, queries
    the model, and returns a greedy valid pair.
    """

    def agent(_obs: Any) -> tuple[int, int]:
        logits, _ = model(make_obs(env.generators, env.pairs))
        return _greedy_pair(np.asarray(logits), env.pairs, len(env.generators))

    return agent


def evaluate_model(
    model: GrobnerPolicyValue,
    env: BuchbergerEnv,
    num_episodes: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Benchmark a model over ``num_episodes`` episodes using greedy actions."""
    return evaluate_agent(
        make_model_agent(model, env), env, num_episodes, base_seed
    )


def plot_episode_rewards(
    rewards: dict[str, np.ndarray],
    title: str = "Per-episode rewards",
    filename: str = "episode_rewards.png",
    figs_dir: str | Path = FIGS_DIR,
) -> Figure:
    """Plot per-episode rewards for each agent with a dashed mean line.

    The figure is always written to ``figs_dir / filename`` so benchmark runs
    leave a persistent artefact in the project's ``figs`` folder.

    Args:
        rewards: Mapping from agent label to a 1-D array of per-episode rewards.
        title: Figure title.
        filename: PNG filename written inside ``figs_dir``.
        figs_dir: Output directory; defaults to ``<project>/figs``.

    Returns:
        The created :class:`matplotlib.figure.Figure`.
    """
    if not rewards:
        raise ValueError("rewards must contain at least one agent")

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    for label, values in rewards.items():
        episodes = np.arange(1, len(values) + 1)
        mean = float(np.mean(values))
        (line,) = ax.plot(
            episodes, values, marker="o", label=f"{label} (mean={mean:.2f})"
        )
        ax.axhline(mean, color=line.get_color(), linestyle="--", alpha=0.6)

    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    output_dir = Path(figs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150)

    return fig


def benchmark(
    agents: dict[str, Agent],
    env: BaseEnv,
    num_episodes: int,
    base_seed: int = 0,
    title: str = "Benchmark",
    filename: str = "benchmark.png",
    figs_dir: str | Path = FIGS_DIR,
) -> tuple[dict[str, np.ndarray], Figure]:
    """Benchmark multiple agents on the same seeded episode set and plot results.

    Each agent runs ``num_episodes`` episodes with seeds
    ``base_seed, base_seed + 1, ...``. The plot is saved to
    ``figs_dir / filename``; per-agent reward arrays and the figure are returned.
    """
    results: dict[str, np.ndarray] = {
        label: evaluate_agent(agent, env, num_episodes, base_seed)
        for label, agent in agents.items()
    }
    fig = plot_episode_rewards(
        results, title=title, filename=filename, figs_dir=figs_dir
    )
    return results, fig
