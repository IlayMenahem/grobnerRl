import distrax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def select_action_infrecne(dqn: eqx.Module, obs: jnp.ndarray):
    '''
    selects an action using the DQN model

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (jnp.ndarray): current observation

    returns:
    - action: selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    chosen_action = distrax.Greedy(q_vals).sample(seed=0)

    return chosen_action


@eqx.filter_jit
def select_action(dqn: eqx.Module, obs: jnp.ndarray, epsilon: float, key) -> jnp.ndarray:
    '''
    selects an action using epsilon-greedy policy

    Args:
    - dqn (eqx.Module): the DQN model
    - obs (jnp.ndarray): current observation
    - epsilon (float): exploration rate
    - key (jax.random.PRNGKey): random key for jax

    returns:
    - action: selected action
    '''
    q_vals = eqx.nn.inference_mode(dqn)(obs)
    chosen_action = distrax.EpsilonGreedy(q_vals, epsilon).sample(seed=key)

    return chosen_action
