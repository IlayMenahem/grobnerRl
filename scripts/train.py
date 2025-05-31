import jax
import optax

from grobnerRl.envs.deepgroebner import BuchbergerEnv
from grobnerRl.models import GrobnerExtractor, GrobnerPolicy, GrobnerCritic
from grobnerRl.rl.a2c import train_a2c

if __name__ == "__main__":
    num_episodes = 100
    n_steps = 32
    gamma = 0.99
    seed = 0

    key = jax.random.key(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    policy = GrobnerPolicy(GrobnerExtractor(4, 16, 32, 2, 2, 2, 2, subkey1))
    critic = GrobnerCritic(GrobnerExtractor(4, 16, 32, 2, 2, 2, 2, subkey2))
    env = BuchbergerEnv('3-10-5-uniform', mode='jax')

    optimizer_policy = optax.adam(1e-5)
    optimizer_critic = optax.adam(1e-4)

    train_a2c(env, policy, critic, optimizer_policy, optimizer_critic, gamma, num_episodes, n_steps, key)
