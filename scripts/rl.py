import torch
from grobnerRl.envs.deepgroebner import BuchbergerEnv
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic
from implementations.ppo import train_ppo


if __name__ == "__main__":
    torch.manual_seed(42)
    num_vars = 3
    max_degree = 4
    num_polynomials = 4

    actor_path = 'models/imitation_policy.pth'
    critic_path = 'models/imitation_critic.pth'

    env = BuchbergerEnv(f'{num_vars}-{max_degree}-{num_polynomials}-uniform', mode='train')
    monoms_embedding_dim = 32
    polys_embedding_dim = 64
    ideal_depth = 2
    ideal_num_heads = 2
    extractor_params = (num_vars, monoms_embedding_dim, polys_embedding_dim, ideal_depth, ideal_num_heads)

    actor = GrobnerPolicy(Extractor(*extractor_params))
    critic = GrobnerCritic(Extractor(*extractor_params))

    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))

    optimizer_actor = torch.optim.Adam(actor.parameters(), 1e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), 3e-4)

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

    actor, critic = train_ppo(env, actor, critic, optimizer_actor, optimizer_critic, batch_size, num_epochs, gamma, clip_epsilon, gae_lambda, entropy_coeff, value_loss_coeff, clip_range_vf, target_kl, max_grad_norm, device)

    # actor, critic = train_a2c(env, actor, critic, optimizer_actor, optimizer_critic, gamma, num_steps, batch_size)
