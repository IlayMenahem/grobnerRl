import jax.numpy as jnp
import jax
import optax
import equinox as eqx

@eqx.filter_jit
def dqn_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    '''
    computes the loss for the DQN

    Args:
    - q_network (eqx.Module): the DQN model
    - target_network (eqx.Module): the target DQN model
    - gamma (float): discount factor
    - batch (dict[str, jnp.ndarray]): batch of samples from the replay buffer

    Returns:
    - loss (jnp.ndarray): computed loss
    '''

    obs = batch['obs']
    next_obs = batch['next_obs']
    actions = batch['acts']
    rewards = batch['rews']
    done = batch['done']

    q_vals = eqx.filter_vmap(q_network)(obs)
    q_curr = jnp.take_along_axis(q_vals, actions[:, None], axis=-1).squeeze(-1)
    q_next = jnp.max(eqx.filter_vmap(target_network)(next_obs), axis=-1)

    mask = jnp.where(done, 0.0, 1.0)
    target = jax.lax.stop_gradient(rewards + gamma * q_next * mask)

    losses = optax.losses.huber_loss(q_curr, target)
    loss = jnp.mean(losses)

    return loss


@eqx.filter_jit
def double_dqn_loss(q_network: eqx.Module, target_network: eqx.Module, gamma: float, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    '''
    computes the loss for the Double DQN

    Args:
    - q_network (eqx.Module): the DQN model
    - target_network (eqx.Module): the target DQN model
    - gamma (float): discount factor
    - batch (dict[str, jnp.ndarray]): batch of samples from the replay buffer

    Returns:
    - loss (jnp.ndarray): computed loss
    '''

    obs = batch['obs']
    next_obs = batch['next_obs']
    actions = batch['acts']
    rewards = batch['rews']
    done = batch['done']

    q_vals = eqx.filter_vmap(q_network)(obs)
    q_curr = jnp.take_along_axis(q_vals, actions[:, None], axis=-1).squeeze(-1)

    target_q_next = eqx.filter_vmap(target_network)(next_obs)
    online_actions_curr = jnp.argmax(q_vals, axis=-1)
    q_next = target_q_next[jnp.arange(target_q_next.shape[0]), online_actions_curr]

    mask = jnp.where(done, 0.0, 1.0)
    target = jax.lax.stop_gradient(rewards + gamma * q_next * mask)

    losses = optax.losses.huber_loss(q_curr, target)
    loss = jnp.mean(losses)

    return loss
