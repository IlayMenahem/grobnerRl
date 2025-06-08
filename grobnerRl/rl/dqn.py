import random
import numpy as np
import torch
from torch import nn
from collections import namedtuple
from tqdm import tqdm
import gymnasium as gym

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
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
    states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)
    weights = torch.tensor(weights).unsqueeze(1).to(device)

    q_values = q_net(states).gather(1, actions)

    next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
    next_q_values = target_net(next_states).gather(1, next_actions).detach()

    target = rewards + gamma * next_q_values * (1 - dones)
    td_errors = target - q_values
    loss = (weights * td_errors.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_priorities = td_errors.abs().detach().cpu().numpy().squeeze() + 1e-6
    replay_buffer.update_priorities(indices, new_priorities)

    return loss.item()


def epsilon_schedule(step: int, start: float = 1.0, end: float = 0.1, decay_steps: int = 10000) -> float:
    if step < decay_steps:
        return start - (start - end) * (step / decay_steps)
    else:
        return end


def train_dqn(env, replay_buffer, target_update_freq: int, gamma: float, q_network,
              optimizer, num_steps: int, batch_size: int):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q_network = q_network.to(device)
    target_network = type(q_network)().to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    state, _ = env.reset()
    state = np.array(state)
    losses = []
    rewards = []
    episode_reward = 0.0

    progress_bar = tqdm(total=num_steps, desc="Training DQN")

    for step in range(num_steps):
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = q_network(state_v)
        epsilon = epsilon_schedule(step)
        action = q_values.argmax(dim=1).item() if random.random() > epsilon else random.randint(0, env.action_space.n - 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = np.array(next_state)

        if done:
            rewards.append(episode_reward)
            episode_reward = 0.0

            progress_bar.set_postfix({'reward': np.mean(rewards[-100:])})

            state, _ = env.reset()
            state = np.array(state)

        if len(replay_buffer.buffer) >= batch_size:
            batch, indices, weights = replay_buffer.sample(batch_size)
            loss = compute_ddqn_loss(batch, indices, weights, q_network, target_network, gamma, optimizer, replay_buffer, device)
            losses.append(loss)

        if step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        progress_bar.update(1)

    return losses


if __name__ == '__main__':
    env = gym.make('CartPole-v1')


    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, env.action_space.n)
            )

        def forward(self, x):
            return self.net(x)

    q_net = QNetwork()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
    replay_buffer = PrioritizedReplayBuffer(10000)

    losses = train_dqn(env, replay_buffer, 1000, 0.99, q_net, optimizer, 100000, 256)
