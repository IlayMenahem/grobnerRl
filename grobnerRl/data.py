import os
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from grobnerRl.benchmark.optimalReductions import optimal_reductions
from grobnerRl.envs.ideals import IdealGenerator
from grobnerRl.envs.deepgroebner import make_obs
from grobnerRl.Buchberger.BuchbergerIlay import init, step


def get_optimal_sequence(ideal_generator: IdealGenerator, step_limit: int):
    optimal_sequence = None
    ideal = None

    while optimal_sequence is None:
        ideal = next(ideal_generator)
        optimal_sequence, final_basis, num_steps = optimal_reductions(ideal, step_limit)

    return ideal, optimal_sequence


def generate_data(ideal_generator: IdealGenerator, step_limit: int, size: int, path: str):
    '''
    Generate dataset and stores to a file loadable by a pytorch DataLoader.

    Args:
    - ideal_generator (IdealGenerator): An instance that yields random polynomial ideals.
    - step_limit (int): Maximum number of steps for optimal_reductions search.
    - size (int): Number of (state, action) pairs to generate.
    - path (str): Path to save the generated dataset.

    Returns:
    - None: The function does not return anything but generates and stores the dataset.
    '''
    states: list[tuple[list[np.ndarray], list[tuple[int, int]]]] = []
    actions: list[int] = []

    pbar = tqdm(total=size, desc='Generating dataset', unit='pair')

    while len(states) < size:
        ideal, optimal_sequence = get_optimal_sequence(ideal_generator, step_limit)

        pairs, basis = init(ideal)

        pbar.update(len(optimal_sequence))

        for action in optimal_sequence:
            current_state = make_obs(basis, pairs)

            i, j = action
            flat_action = i * len(current_state[0]) + j

            states.append(current_state)
            actions.append(flat_action)

            basis, pairs = step(basis, pairs, action)

    # save the dataset to the specified path
    dataset = {
        'states': states,
        'actions': actions
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(dataset, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


class BCDataset(Dataset):
    def __init__(self, path: str):
        with open(path, 'r') as f:
            dataset = json.load(f)

        self.states = dataset['states']
        self.actions = dataset['actions']

        states = []
        for state in self.states:
            ideal, pairs = state

            ideal = [np.array(poly) for poly in ideal]
            pairs = [tuple(pair) for pair in pairs]

            states.append((ideal, pairs))

        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]

        return state, action


def bc_collate(batch):
    states, actions = map(list, zip(*batch))

    return states, actions
