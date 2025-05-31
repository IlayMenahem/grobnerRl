from typing import Iterator
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from grobnerRl.benchmark.optimalReductions import optimal_reductions
from grobnerRl.envs.ideals import random_ideal
from grobnerRl.models import make_obs
from grobnerRl.Buchberger.BuchbergerIlay import init, step
from grobnerRl.rl.utils import update_network


def bc_loss(policy: eqx.Module, actions, states):
    def loss_fn(state, action):
        action_pred = policy(state)
        action_one_hot = jnp.zeros_like(action_pred).at[action].set(1.0)
        loss = optax.softmax_cross_entropy(action_pred, action_one_hot, axis=None)

        return loss

    losses = jax.tree.map(loss_fn, states, actions)
    loss = jnp.mean(jnp.array(losses))

    return loss


def train_bc(policy: eqx.Module, dataloader, num_steps, optimizer):
    optimizer_state = optimizer.init(policy)
    progress_bar = tqdm(total=num_steps, unit="episode")

    for _ in range(num_steps):
        states, actions = next(dataloader)
        policy, loss, optimizer_state = update_network(policy, optimizer, optimizer_state, bc_loss, actions, states)

        progress_bar.set_postfix(loss=loss)
        progress_bar.update(1)

    progress_bar.close()

    return policy


class BCDataloader:
    """
    Dataloader for behavioral cloning that generates expert demonstrations
    using optimal_reductions and collects (state, action) pairs.

    This dataloader generates random polynomial ideals, computes optimal
    reduction sequences using A* search, and simulates the Buchberger
    algorithm to collect training data for behavioral cloning.

    Example:
        >>> import sympy as sp
        >>> ideal_params = (3, 5, 3, 3, sp.FF(32003), 'grevlex')
        >>> dataloader = BCDataloader(ideal_params, batch_size=16)
        >>> states, actions = next(dataloader)
        >>> print(f"Batch size: {len(states)}")
    """

    def __init__(self, ideal_params: tuple, step_limit: int, batch_size: int):
        """
        Initialize the BC dataloader.

        Args:
            ideal_params: Parameters for random_ideal generation
                         (num_polys, max_num_monoms, max_degree, num_vars, field, order)
            step_limit: Maximum number of steps for optimal_reductions search
            batch_size: Number of (state, action) pairs per batch

        Raises:
            ValueError: If ideal_params has incorrect length or invalid parameters
        """
        if len(ideal_params) != 6:
            raise ValueError("ideal_params must have 6 elements: (num_polys, max_num_monoms, max_degree, num_vars, field, order)")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if step_limit <= 0:
            raise ValueError("step_limit must be positive")

        self.ideal_params = ideal_params
        self.step_limit = step_limit
        self.batch_size = batch_size

    def generate_batch(self) -> tuple[list, list]:
        states = []
        actions = []

        while True:
            optimal_sequence = None

            while optimal_sequence is None:
                ideal = random_ideal(*self.ideal_params)
                optimal_sequence, final_basis, num_steps = optimal_reductions(ideal, self.step_limit)

            pairs, basis = init(ideal)

            for action in optimal_sequence:
                if len(states) >= self.batch_size:
                    return states, actions

                current_state = make_obs(basis, pairs)
                states.append(current_state)
                actions.append(action)

                basis, pairs = step(basis, pairs, action)

        return [], []

    def __iter__(self) -> Iterator[tuple[list, list]]:
        while True:
            yield self.generate_batch()

    def __next__(self) -> tuple[list, list]:
        return self.generate_batch()
