import jax
import jax.numpy as jnp
import equinox as eqx
from chex import Array


def select_action_infrecne(dqn: eqx.Module, obs: Array) -> tuple[int, int]:
    '''
    selects an action using the DQN model

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (Array): current observation

    returns:
    - action (tuple[int, int]): selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    chosen_action = jnp.array(jnp.unravel_index(jnp.argmax(q_vals), q_vals.shape))

    i, j = chosen_action
    chosen_action = (i.item(), j.item())

    return chosen_action


def uniform_sample_index(key: Array, mask: Array):
    '''
    samples an array index uniformly from the array

    Args:
    - key (Array): random key for jax
    - mask (Array): mask of legal indices

    returns:
    - result (Array | int): sampled index
    '''
    flat_mask = mask.reshape(-1)
    flat_length = len(flat_mask)
    flat_indices = jnp.arange(flat_length)[flat_mask]
    sampled_index = jax.random.choice(key, flat_indices)
    sampled_index = jnp.unravel_index(sampled_index, mask.shape)
    sampled_index = jnp.array(sampled_index)

    return sampled_index


def select_action(dqn: eqx.Module, obs: Array, epsilon: float, key: Array) -> tuple[int, int]:
    '''
    selects an action using epsilon-greedy policy

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (Array): current observation, can be n dimensional
    - epsilon (float): exploration rate
    - key (Array): random key for jax

    returns:
    - action (tuple[int, int]): selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    is_legal = jnp.isfinite(q_vals)

    greedy_action = jnp.array(jnp.unravel_index(jnp.argmax(q_vals), q_vals.shape))
    epsilon_action = uniform_sample_index(key, is_legal)
    chosen_action = jax.lax.select(jax.random.uniform(key) < epsilon, epsilon_action, greedy_action)

    i, j = chosen_action
    chosen_action = (i.item(), j.item())

    return chosen_action
