import torch
from grobnerRl.envs.deepgroebner import BuchbergerEnv
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic, GrobnerEvaluator
from implementations.ppo import train_ppo
from implementations.a2c import train_a2c
from implementations.dqn import train_dqn, PrioritizedReplayBuffer

if __name__ == "__main__":
    num_vars = 3
    max_degree = 4
    num_polynomials = 4

    env = BuchbergerEnv(f'{num_vars}-{max_degree}-{num_polynomials}-uniform', mode='train')
    extractor_args = (num_vars+1, 32, 64, 4, 4)

    actor = GrobnerPolicy(Extractor(*extractor_args))
    critic = GrobnerCritic(num_vars+1, 32, 64, 1024)
    evaluator = GrobnerEvaluator(Extractor(*extractor_args))

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)
    optimizer_evaluator = torch.optim.Adam(evaluator.parameters(), lr=1e-4)

    batch_size = 1024
    num_epochs = 250
    gamma = 0.99

    clip_epsilon = 0.02
    gae_lambda = 0.97
    entropy_coeff = 0.001
    value_loss_coeff = 0.9
    clip_range_vf = 0.01
    target_kl = 1.0
    max_grad_norm = 0.1

    replay_buffer = PrioritizedReplayBuffer(50000)
    target_update_freq = 1000

    actor, critic = train_ppo(env, actor, critic, optimizer_actor, optimizer_critic,
        batch_size, num_epochs, gamma, clip_epsilon, gae_lambda, entropy_coeff,
        value_loss_coeff, clip_range_vf, target_kl, max_grad_norm)

    actor, critic = train_a2c(env, actor, critic, optimizer_actor, optimizer_critic, gamma, num_epochs, batch_size)
