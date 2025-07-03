import random
import numpy as np
import torch
from torch import nn
from collections import namedtuple
from tqdm import tqdm
import gymnasium as gym
import copy

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,))
        self.frame = 1

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def beta_by_frame(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        beta = self.beta_by_frame()
        self.frame += 1
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority


def compute_ddqn_loss(batch, indices, weights, q_net, target_net, gamma, optimizer, replay_buffer, device):
    states, action, reward, next_states, done = batch

    actions = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(reward).unsqueeze(1).to(device)
    dones = torch.tensor(done).unsqueeze(1).to(device)
    weights = torch.tensor(weights).unsqueeze(1).to(device)

    q_values = q_net(states).gather(1, actions)

    next_actions = q_net(next_states).argmax(dim=-1, keepdim=True)
    next_q_values = target_net(next_states).gather(1, next_actions).detach()

    target = rewards + gamma * next_q_values * ~dones
    td_errors = target - q_values
    loss = (weights * td_errors.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_priorities = td_errors.abs().detach().cpu().numpy().squeeze() + 1e-6
    replay_buffer.update_priorities(indices, new_priorities)

    return loss.item()


def epsilon_schedule(step: int, start: float = 1.0, end: float = 0.05, decay_steps: int = 10000) -> float:
    if step < decay_steps:
        return start - (start - end) * (step / decay_steps)
    else:
        return end


def train_dqn(env, replay_buffer, target_update_freq: int, gamma: float, q_network: nn.Module,
              optimizer, num_steps: int, batch_size: int, device=torch.device('cpu')):

    q_network = q_network.to(device)
    target_network = copy.deepcopy(q_network).to(device)
    target_network.eval()

    state, _ = env.reset()
    losses = []
    rewards = []
    episode_reward = 0.0

    progress_bar = tqdm(total=num_steps, desc="Training DQN")

    for step in range(num_steps):
        q_values = q_network(state)
        epsilon = epsilon_schedule(step)
        action = q_values.argmax(dim=-1).item() if random.random() > epsilon else random.randint(0, len(q_values) - 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            rewards.append(episode_reward)
            episode_reward = 0.0

            progress_bar.set_postfix({'reward': np.mean(rewards[-100:])})

            state, _ = env.reset()

        if len(replay_buffer.buffer) >= batch_size:
            batch, indices, weights = replay_buffer.sample(batch_size)
            loss = compute_ddqn_loss(batch, indices, weights, q_network, target_network, gamma, optimizer, replay_buffer, device)
            losses.append(loss)

        if step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        progress_bar.update(1)

    return losses, rewards, q_network


if __name__ == '__main__':
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.advantage = nn.Linear(128, action_dim)
            self.value = nn.Linear(128, 1)

        def forward(self, x):
            if isinstance(x, list):
                return self.forward(torch.tensor(np.array(x)).to(self.fc1.weight.device))

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(np.array(x)).to(self.fc1.weight.device)

            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))

            # use dueling architecture
            advantage = self.advantage(x)
            value = self.value(x)
            x = value + advantage - advantage.mean(dim=-1, keepdim=True)

            return x


    env = gym.make('CartPole-v1', max_episode_steps=2048)
    device = torch.device('cpu')

    q_net = QNetwork(4, 2)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = PrioritizedReplayBuffer(10000)

    losses = train_dqn(env, replay_buffer, 1000, 0.99, q_net, optimizer, 250000, 512, device)
