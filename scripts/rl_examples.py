import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import gymnasium as gym
from grobnerRl.rl.a2c import train_a2c
from grobnerRl.rl.dqn import train_dqn, ReplayBuffer, dqn_loss

class agent(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    state_value: eqx.nn.Linear
    advantage: eqx.nn.Linear

    def __init__(self, input_size: int, output_size: int, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.linear1 = eqx.nn.Linear(input_size, 32, key=key1)
        self.linear2 = eqx.nn.Linear(32, 32, key=key2)
        self.linear3 = eqx.nn.Linear(32, 32, key=key3)
        self.linear4 = eqx.nn.Linear(32, 32, key=key4)
        self.state_value = eqx.nn.Linear(32, 1, key=key5)
        self.advantage = eqx.nn.Linear(32, output_size, key=key6)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))

        state_value = self.state_value(x)
        advantage = self.advantage(x)

        q_values = state_value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))

        return q_values


class policy(eqx.Module):
    value: agent

    def __init__(self, input_size: int, output_size: int, key):
        self.value = agent(input_size, output_size, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        vals = self.value(x)
        probs = jax.nn.softmax(vals)

        return probs

def a2c_example():
    num_episodes = 2500
    n_steps = 32
    gamma = 0.99
    seed = 0
    key = jax.random.key(seed)

    env = gym.make('CartPole-v1', max_episode_steps=500)
    actor = policy(4, 2, key)
    critic = agent(4, 1, key)

    optimizer_policy = optax.adam(1e-4)
    optimizer_policy_state = optimizer_policy.init(actor)
    optimizer_critic = optax.adam(3e-4)
    optimizer_critic_state = optimizer_critic.init(critic)

    actor, critic, scores, losses = train_a2c(env, actor, critic, optimizer_policy,
        optimizer_policy_state, optimizer_critic, optimizer_critic_state, gamma,
        num_episodes, n_steps, key)

    env.close()


def dqn_example():
    num_steps = 50000
    gamma = 0.99
    seed = 0
    target_update_freq = 250

    capacity = 20000
    batch_size = 512

    initial_epsilon = 1.0
    transition_steps = 20000
    final_epsilon = 0.1
    learning_rate = 4e-5
    max_norm = 0.25

    key = jax.random.key(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    q_network = agent(6, 3, subkey1)
    target_network = agent(6, 3, subkey2)
    env = gym.make('Acrobot-v1', max_episode_steps=250)

    replay_buffer = ReplayBuffer(capacity, batch_size)

    epsilon_shed = optax.schedules.linear_schedule(initial_epsilon, final_epsilon,
        transition_steps, batch_size)

    optimizer = optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
    optimizer_state = optimizer.init(q_network)

    train_dqn(env, replay_buffer, epsilon_shed, target_update_freq, gamma, q_network,
        target_network, optimizer, optimizer_state, num_steps, dqn_loss, key)

    env.close()

if __name__ == "__main__":
    a2c_example()
