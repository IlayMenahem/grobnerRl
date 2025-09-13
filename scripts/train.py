import torch
from grobnerRl.envs.deepgroebner import BuchbergerEnv
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic, GrobnerValue
from implementations.ppo import train_ppo
from implementations.a2c import train_a2c


if __name__ == "__main__":
    num_vars = 3
    max_degree = 20
    num_polynomials = 4

    env = BuchbergerEnv(f'{num_vars}-{max_degree}-{num_polynomials}-uniform', mode='train')
    extractor_args = (num_vars, 16, 32, 2, 2)

    actor = GrobnerPolicy(Extractor(*extractor_args))
    critic = GrobnerCritic(Extractor(*extractor_args))

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    batch_size = 2048
    num_epochs = 1000
    num_steps = batch_size * num_epochs

    gamma = 0.99
    gae_lambda = 0.97
    clip_epsilon = 0.2
    entropy_coeff = 0.01
    value_loss_coeff = 0.5
    clip_range_vf = 0.01
    target_kl = 0.01
    max_grad_norm = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    actor, critic = train_a2c(env, actor, critic, optimizer_actor, optimizer_critic, gamma, num_steps, batch_size)
    actor, critic = train_ppo(env, actor, critic, optimizer_actor, optimizer_critic, batch_size, num_epochs, gamma, clip_epsilon, gae_lambda, entropy_coeff, value_loss_coeff, clip_range_vf, target_kl, max_grad_norm, device)
