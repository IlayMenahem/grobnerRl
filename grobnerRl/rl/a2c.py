import torch
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
from grobnerRl.rl.utils import plot_learning_process


def compute_returns(rewards, dones, next_value, gamma, device):
    R = next_value
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - float(d))
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float, device=device)


def rollout(env, policy, critic, n_steps, device):
    log_probs = []
    values = []
    rewards = []
    entropies = []
    dones = []

    state, _ = env.reset()
    done = False

    for _ in range(n_steps):
        probs = policy(state)
        value = critic(state).squeeze(0).squeeze(-1)
        dist = torch.distributions.Categorical(probs=probs.to(device))
        action = dist.sample()
        log_prob = dist.log_prob(action).squeeze(-1)
        entropy = dist.entropy().squeeze(-1)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)
        dones.append(done)

        state = next_state
        if done:
            break

    if done:
        next_value = 0.0
    else:
        next_value = critic(next_state).item()

    return log_probs, values, rewards, entropies, dones, next_value, sum(rewards)


def update_models(optimizer_policy, optimizer_critic, log_probs, values, returns, entropies, entropy_coeff):
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    advantages = returns - values

    policy_loss = - (log_probs * advantages.detach()).sum() - entropy_coeff * entropies.sum()
    critic_loss = F.mse_loss(values, returns)

    optimizer_policy.zero_grad()
    optimizer_critic.zero_grad()

    policy_loss.backward()
    critic_loss.backward()

    optimizer_policy.step()
    optimizer_critic.step()

    return policy_loss.item(), critic_loss.item()


def train_a2c(env, policy, critic, optimizer_policy, optimizer_critic,
              gamma: float, num_episodes: int, n_steps: int, entropy_coeff: float = 0.01):
    device = next(policy.parameters()).device
    episode_rewards = []
    actor_losses = []
    critic_losses = []

    pbar = tqdm(range(num_episodes), desc="Training A2C")
    for _ in pbar:
        log_probs, values, rewards, entropies, dones, next_value, total_reward = rollout(env, policy, critic, n_steps, device)
        returns = compute_returns(rewards, dones, next_value, gamma, device)
        policy_loss, critic_loss = update_models(optimizer_policy, optimizer_critic, log_probs, values, returns, entropies, entropy_coeff)

        episode_rewards.append(total_reward)
        actor_losses.append(policy_loss)
        critic_losses.append(critic_loss)
        avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
        pbar.set_postfix({"avg_reward_100": f"{avg_reward:.2f}"})

    plot_learning_process(episode_rewards, actor_losses, critic_losses)

    return policy, critic


if __name__ == "__main__":
    import torch.nn as nn

    env = gym.make("CartPole-v1", max_episode_steps=1024)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            x = torch.tensor(x, dtype=torch.float32)
            probs = self.fc(x)
            return probs

    class CriticNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            x = torch.tensor(x, dtype=torch.float32)
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet().to(device)
    critic = CriticNet().to(device)
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    train_a2c(env, policy, critic, optimizer_policy, optimizer_critic, 0.99, 1000, 1024, 0.01)
