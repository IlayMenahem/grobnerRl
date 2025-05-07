import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from chex import Array

from .utils import GroebnerState


def td_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, obs: GroebnerState,
    next_obs: GroebnerState, action, reward, done) -> Array:
    q_vals = q_network(obs)
    q_curr = q_vals[action]

    target_q_next = target_network(next_obs)
    q_next = jnp.max(target_q_next)

    mask = jnp.where(done, 0.0, 1.0)
    target = jax.lax.stop_gradient(reward + gamma * q_next * mask)

    loss = optax.losses.huber_loss(q_curr, target)

    return loss


def dqn_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, batch: dict) -> jnp.ndarray:
    '''
    computes the loss for the Double DQN

    Args:
    - q_network (eqx.Module): the DQN model
    - target_network (eqx.Module): the target DQN model
    - gamma (float): discount factor
    - batch (dict): batch of samples from the replay buffer

    Returns:
    - loss (jnp.ndarray): computed loss
    '''

    observations: list[GroebnerState] = batch['obs']
    next_observations: list[GroebnerState] = batch['next_obs']
    actions: list[Array] = batch['acts']
    rewards: list[Array] = batch['rews']
    dones: list[Array] = batch['done']

    def td_wrapper(obs, next_obs, action, reward, done):
        return td_loss(q_network, target_network, gamma, obs, next_obs, action, reward, done)

    losses = jax.tree.map(td_wrapper, observations, next_observations, actions, rewards, dones, is_leaf=lambda x: isinstance(x, GroebnerState))
    loss = jnp.mean(jnp.array(losses))

    return loss
