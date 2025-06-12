import torch
from grobnerRl.envs.deepgroebner import BuchbergerEnv
from grobnerRl.rl.ppo import ppo
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic

if __name__ == "__main__":
    num_vars = 3
    max_degree = 4
    num_polynomials = 4

    env = BuchbergerEnv(f'{num_vars}-{max_degree}-{num_polynomials}-uniform', mode='train')
    extractor_args = (num_vars, 16, 32, 2, 4, 2, 4)

    actor = GrobnerPolicy(Extractor(*extractor_args))
    critic = GrobnerCritic(Extractor(*extractor_args))

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

    batch_size = 1024
    num_epochs = 250
    gamma = 0.99

    actor, critic = ppo(env, actor, critic, optimizer_actor, optimizer_critic,
        batch_size, num_epochs, gamma, clip_epsilon=0.02, gae_lambda=0.97,
        entropy_coeff=0.01, value_loss_coeff=0.5, clip_range_vf=0.02,
        target_kl=1.0, max_grad_norm=0.1)
