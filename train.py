import jax
import jax.numpy as jnp
import optax
from envs.deepgroebner import BuchbergerEnv
from models import GrobnerModel

from rl.buffers import ReplayBuffer
from rl.algorithms import train_dqn
from rl.losses import dqn_loss

if __name__ == "__main__":

    num_steps = 50000
    gamma = 0.99
    seed = 0
    target_update_freq = 125

    capacity = 20000
    batch_size = 16

    initial_epsilon = 1.0
    transition_steps = 20000
    final_epsilon = 0.1
    learning_rate = 4e-5
    max_norm = 0.25

    key = jax.random.key(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    q_network = GrobnerModel(128, 16, 32, 2, 2, 2, 2, -jnp.inf, subkey1)
    target_network = GrobnerModel(128, 16, 32, 2, 2, 2, 2, -jnp.inf, subkey2)
    env = BuchbergerEnv('2-3-3-uniform')

    replay_buffer = ReplayBuffer(capacity, batch_size)

    epsilon_shed = optax.schedules.linear_schedule(initial_epsilon, final_epsilon,
        transition_steps, batch_size)

    optimizer = optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
    optimizer_state = optimizer.init(q_network)

    train_dqn(env, replay_buffer, epsilon_shed, target_update_freq, gamma, q_network,
        target_network, optimizer, optimizer_state, num_steps, dqn_loss, key)
