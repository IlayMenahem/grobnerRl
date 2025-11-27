import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate

import numpy as np
import torch
from grain.sources import RandomAccessDataSource
from torch.utils.data import Dataset
from tqdm import tqdm

from grobnerRl.envs.env import BaseEnv, BuchbergerEnv, tokenize
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.experts import BasicExpert, ClosestLMExpert, Expert
from grobnerRl.types import Action, Observation


class JsonDatasource(RandomAccessDataSource[tuple[Observation, Action]]):
    def __init__(self, path: str, obs: str, labels: str):
        with open(path, "r") as f:
            dataset = json.load(f)

        self.states = dataset[obs]
        self.actions = dataset[labels]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[Observation, Action]:
        state = self.states[idx]
        action = self.actions[idx]

        return state, action


class JsonDataset(Dataset):
    def __init__(self, path: str, obs: str, labels: str):
        with open(path, "r") as f:
            dataset = json.load(f)

        self.states = dataset[obs]
        self.actions = dataset[labels]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]

        return state, action


def collate(batch):
    """
    Collate function that creates nested tensors for variable-length polynomial ideals.

    Each state is a tuple (ideal, selectables) where:
    - ideal: list of polynomials (each polynomial is a numpy array of monomials)
    - selectables: list of tuples indicating valid actions

    Returns:
    - nested_batch: List of nested tensors (one per batch item), representing the batch
    - selectables_batch: List of selectables for each batch item
    - actions_tensor: Tensor of actions
    """
    states, actions = map(list, zip(*batch))

    nested_batch = []
    selectables_batch = []

    for state in states:
        ideal, pairs = state

        pairs = [tuple(pair) for pair in pairs]

        poly_tensors = [torch.from_numpy(np.array(poly)).float() for poly in ideal]
        nested_ideal = torch.nested.nested_tensor(poly_tensors, layout=torch.jagged)

        nested_batch.append(nested_ideal)
        selectables_batch.append(pairs)

    actions_tensor = torch.tensor(actions, dtype=torch.long)

    return (nested_batch, selectables_batch), actions_tensor


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

                states.append((tokenize(obs[0]), obs[1]))
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


def _generate_episode_data(args):
    """Helper function to generate data from a single episode in a separate process."""
    ideal_dist, n_simulations, c, gamma_agent, rollout_policy = args

    # Import here to avoid issues with multiprocessing
    from grobnerRl.envs.deepgroebner import BuchbergerEnv, MCTSAgent

    # Create environment and agent in the worker process
    env = BuchbergerEnv(ideal_dist=ideal_dist, mode="train")
    expert_agent = MCTSAgent(
        env,
        n_simulations=n_simulations,
        c=c,
        gamma=gamma_agent,
        rollout_policy=rollout_policy,
    )

    states = []
    actions = []
    rewards = []

    obs, _ = env.reset()
    episode_done = False

    while not episode_done:
        expert_action = expert_agent.act((env.G, env.P))

        # Convert expert action to flat index
        i, j = expert_action
        flat_action = i * len(obs[0]) + j

        states.append(obs)
        actions.append(flat_action)

        obs, reward, terminated, truncated, info = env.step(expert_action)
        rewards.append(reward)
        episode_done = terminated or truncated or info.get("invalid_action", False)

    return states, actions, rewards


def generate_expert_data_concurrent(
    ideal_dist,
    size,
    path,
    n_simulations=50,
    c=1.0,
    gamma=0.99,
    rollout_policy="normal",
    num_workers=4,
):
    """Generate expert demonstrations concurrently using multiple environments.

    Parameters
    ----------
    ideal_dist : str
        Ideal distribution specification.
    size : int
        Total number of state-action pairs to generate.
    path : str
        Path to save the dataset.
    n_simulations : int, optional
        Number of MCTS simulations per action.
    c : float, optional
        Exploration constant for MCTS.
    gamma : float, optional
        Discount factor for computing values.
    rollout_policy : str, optional
        Policy for MCTS rollouts ('random', 'normal', 'degree', 'first').
    num_workers : int, optional
        Number of parallel worker processes.

    Returns
    -------
    str
        Path to the saved dataset.
    """
    all_states = []
    all_actions = []
    all_values = []

    worker_args = (ideal_dist, n_simulations, c, gamma, rollout_policy)

    with tqdm(total=size, desc="Generating expert data", unit="pairs") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_episode_data, worker_args)
                for _ in range(num_workers)
            }

            while futures and len(all_states) < size:
                done_futures = set()
                for future in as_completed(futures):
                    done_futures.add(future)
                    break  # Process one at a time

                # Remove completed future from pending set
                futures -= done_futures

                for done in done_futures:
                    states, actions, rewards = done.result()

                    # Compute discounted returns for this episode
                    states_values = list(
                        reversed(
                            list(
                                accumulate(
                                    reversed(rewards), lambda acc, r: r + gamma * acc
                                )
                            )
                        )
                    )

                    # Add data from this episode, respecting the size limit
                    pairs_to_add = min(len(states), size - len(all_states))
                    all_states.extend(states[:pairs_to_add])
                    all_actions.extend(actions[:pairs_to_add])
                    all_values.extend(states_values[:pairs_to_add])

                    pbar.update(pairs_to_add)

                    if len(all_states) < size:
                        futures.add(
                            executor.submit(_generate_episode_data, worker_args)
                        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = {"states": all_states, "actions": all_actions, "values": all_values}
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
        data_source=datasource, sampler=sampler, operations=(to_batch,), worker_count=0
    )

    num_vars = 4
    num_clauses = int(num_vars * 4.5)
    size = 1000
    ideal_generator = SAT3IdealGenerator(num_vars, num_clauses)
    env = BuchbergerEnv(ideal_generator)
    expert = BasicExpert(env)

    path = os.path.join("data", f"sat_{num_vars}-{num_clauses}.json")
    generate_expert_data(env, size, path, expert)
