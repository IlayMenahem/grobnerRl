import json
import os
from collections.abc import Sequence
from itertools import accumulate
from typing import SupportsIndex, Union

import numpy as np
from grain.sources import RandomAccessDataSource
from tqdm import tqdm

from grobnerRl.env import BaseEnv, BuchbergerEnv, tokenize
from grobnerRl.ideals import SAT3IdealGenerator
from grobnerRl.experts import BasicExpert, Expert
from grobnerRl.types import Action, Observation


class JsonDatasource(RandomAccessDataSource[Union[tuple[Observation, Action], tuple]]):
    def __init__(
        self,
        path: str,
        obs: str,
        labels: Sequence[str],
        indices: Sequence[int] | None = None,
    ):
        with open(path, "r") as f:
            dataset = json.load(f)

        self.states = dataset[obs]
        self.labels = labels

        # Load all label arrays
        self.label_data = [dataset[label] for label in self.labels]

        if indices is not None:
            self.states = [self.states[int(i)] for i in indices]
            self.label_data = [
                [label_array[int(i)] for i in indices]
                for label_array in self.label_data
            ]

    def __len__(self):
        return len(self.states)

    def __getitem__(
        self, idx: SupportsIndex
    ) -> Union[tuple[Observation, Action], tuple]:
        state = self.states[idx]
        label_values = [label_data[idx] for label_data in self.label_data]

        return (state, *label_values)


def generate_expert_data(env: BaseEnv, size, path, expert_agent: Expert, gamma=0.99):
    """Generate expert demonstrations using provided expert agent."""
    states = []
    actions = []
    values = []

    with tqdm(total=size, desc="Generating expert data", unit="pairs") as pbar:
        while len(states) < size:
            obs, _ = env.reset()
            expert_agent.update_env(env)
            episode_done = False
            rewards = []

            if not obs[1]:  # If the ideal is empty, skip this episode
                continue

            while not episode_done and len(states) < size:
                expert_action = expert_agent(obs)

                # Convert expert action to flat index
                i, j = expert_action
                flat_action = i * len(obs[0]) + j

                states.append((tokenize(obs[0]), list(obs[1])))
                actions.append(flat_action)
                pbar.update(1)

                obs, reward, terminated, truncated, info = env.step(expert_action)
                rewards.append(reward)
                episode_done = (
                    terminated or truncated or info.get("invalid_action", False)
                )

            states_values = reversed(
                list(accumulate(reversed(rewards), lambda acc, r: r + gamma * acc))
            )
            values.extend(states_values)

    # Save dataset
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = {"states": states, "actions": actions, "values": values}
    with open(path, "w") as f:
        json.dump(
            dataset, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

    return path


if __name__ == "__main__":
    from grain import DataLoader
    from grain.samplers import IndexSampler
    from grain.sharding import ShardOptions
    from grain.transforms import Batch

    def batch_fn(
        x: Sequence[tuple[Observation, Action]],
    ) -> tuple[Sequence[Observation], Sequence[Action]]:
        observations, actions = zip(*x)

        return observations, actions

    batch_size = 4
    to_batch = Batch(batch_size, True, batch_fn)

    ideal_dist = "3-10_sat3"
    data_path = os.path.join("data", f"{ideal_dist}.json")
    datasource = JsonDatasource(data_path, "states", "actions")
    sampler = IndexSampler(len(datasource), ShardOptions(0, 1, True), True, seed=0)
    dataloader = DataLoader(
        data_source=datasource, sampler=sampler, operations=(to_batch,), worker_count=2
    )

    num_vars = 4
    num_clauses = int(num_vars * 4.5)
    size = 1000
    ideal_generator = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_generator)
    expert = BasicExpert(env)

    path = os.path.join("data", f"sat_{num_vars}-{num_clauses}.json")
    generate_expert_data(env, size, path, expert)
