import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm


def compute_gae(rewards: list[float], values: list[float], dones: list[bool], gamma: float, gae_lambda: float) -> list[float]:
    advantages = []
    gae = 0
    values = values + [0]

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)

    return advantages


def ppo_update(actor: nn.Module, critic: nn.Module, optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer, states: list, actions: torch.Tensor, old_log_probs: torch.Tensor,
    returns: torch.Tensor, advantages: torch.Tensor, clip_epsilon: float, entropy_coeff: float, value_loss_coeff: float,
    clip_range_vf: float, max_grad_norm: float, target_kl: float) -> tuple[float, float, float, float]:

    actions = torch.tensor(actions)
    old_log_probs = torch.tensor(old_log_probs)
    returns = torch.tensor(returns)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    probs = actor(states)
    dist = Categorical(probs=probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_pred = critic(states).squeeze()
    value_pred_clipped = returns + (value_pred - returns).clamp(-clip_range_vf, clip_range_vf)
    value_losses = (value_pred - returns).pow(2)
    value_losses_clipped = (value_pred_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    loss = policy_loss + value_loss * value_loss_coeff - entropy * entropy_coeff
    loss.backward()

    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

    optimizer_actor.step()
    optimizer_critic.step()

    approx_kl = (old_log_probs - new_log_probs).mean().item()

    return policy_loss.item(), value_loss.item(), entropy.item(), approx_kl


def collect_trajectories(env: gym.Env, actor: nn.Module, critic: nn.Module, batch_size: int, gamma: float) -> tuple[list, list[int], list[float], list[bool], list[float], list[float], float]:
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    episode_rewards: list[float] = []
    ep_reward: float = 0

    state, _ = env.reset()
    done: bool = False

    pbar = tqdm(total=batch_size, desc="Collecting trajectories", leave=False)
    for _ in range(batch_size):
        probs = actor(state)
        dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob: float = dist.log_prob(action).item()
        value: float = critic(state).item()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        ep_reward += float(reward)
        state = next_state
        pbar.update(1)

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            state, _ = env.reset()
            done = False

    pbar.close()
    avg_reward: float = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

    return states, actions, rewards, dones, log_probs, values, avg_reward


def train_ppo(env: gym.Env, actor: nn.Module, critic: nn.Module, optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer, batch_size: int, num_epochs: int, gamma: float,
    clip_epsilon: float, gae_lambda: float, entropy_coeff: float, value_loss_coeff: float,
    clip_range_vf: float, target_kl: float, max_grad_norm: float)-> tuple[nn.Module, nn.Module]:

    for epoch in tqdm(range(num_epochs), desc="PPO epochs"):
        states, actions, rewards, dones, old_log_probs, values, avg_reward = collect_trajectories(env, actor, critic, batch_size, gamma)
        advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
        returns = [adv + val for adv, val in zip(advantages, values)]

        policy_loss, value_loss, entropy, approx_kl = ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                                                                 states, actions, old_log_probs, returns, advantages,
                                                                 clip_epsilon, entropy_coeff, value_loss_coeff,
                                                                 clip_range_vf, max_grad_norm, target_kl)
        tqdm.write(f"Epoch {epoch + 1}: Avg Reward {avg_reward:.2f}  Policy Loss {policy_loss:.10f}  Value Loss {value_loss:.3f}  Entropy {entropy:.3f}  KL {approx_kl:.6f}")

        if approx_kl > target_kl:
            print(f"Early stopping at epoch {epoch + 1} due to reaching target KL (KL: {approx_kl:.6f} > {target_kl})")
            break

    return actor, critic


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        if isinstance(x, list):
            return self.forward(torch.tensor(np.array(x)))

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        probs = torch.softmax(x, dim=-1)

        return probs


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        if isinstance(x, list):
            return self.forward(torch.tensor(np.array(x)))

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=1024)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    train_ppo(env, actor, critic, optimizer_actor, optimizer_critic, 4096, 250,
        gamma=0.99, clip_epsilon=0.2, gae_lambda=0.95, entropy_coeff=0.01,
        value_loss_coeff=0.5, clip_range_vf=0.2, target_kl=0.03, max_grad_norm=0.5)
