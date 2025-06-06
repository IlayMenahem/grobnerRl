import sympy
import optax
import jax

from grobnerRl.models import GrobnerPolicy, GrobnerExtractor
from grobnerRl.rl.bc import train_bc, BCDataloader

if __name__ == "__main__":
    ideal_params = (10, 3, 5, 3, sympy.FF(32003), 'grevlex')
    dataloader = BCDataloader(ideal_params, 10000, 256)

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    policy = GrobnerPolicy(GrobnerExtractor(3, 16, 32, 1, 1, 1, 1, subkey))

    optimizer = optax.chain(optax.clip_by_global_norm(0.25), optax.adam(1e-4))
    policy = train_bc(policy, dataloader, 100, optimizer)
