from typing import Dict, List, Tuple, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    ptr: int = 0
    size: int = 0

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        batch = dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs], acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs], done=self.done_buf[idxs])

        return batch

    def can_sample(self) -> bool:
        return self.size >= self.batch_size


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x)+x)
        x = self.linear3(x)

        return x


def seed_torch(seed):
    torch.manual_seed(seed)

    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def _compute_dqn_loss(samples: Dict[str, np.ndarray], dqn: Network, dqn_target: Network,
    gamma: float, device: torch.device) -> torch.Tensor:
    '''
    Computes the DQN loss based on the sampled experiences.

    Args:
    samples (Dict[str, np.ndarray]): A dictionary containing the sampled experiences.
    dqn (Network): The local DQN network.
    dqn_target (Network): The target DQN network.
    gamma (float): The discount factor.
    device (torch.device): The device to perform computations on.

    Returns:
    torch.Tensor: The computed DQN loss.
    '''
    state = torch.FloatTensor(samples["obs"]).to(device)
    next_state = torch.FloatTensor(samples["next_obs"]).to(device)
    action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    curr_q_value = dqn(state).gather(1, action)
    with torch.no_grad():
        next_q_value = dqn_target(next_state).max(dim=1, keepdim=True)[0]

    mask = 1 - done
    target = (reward + gamma * next_q_value * mask).to(device)

    loss = F.smooth_l1_loss(curr_q_value, target)

    return loss

def _target_hard_update(dqn: Network, dqn_target: Network) -> None:
    '''
    Hard update of the target network parameters.

    Args:
    dqn (Network): The local DQN network.
    dqn_target (Network): The target DQN network.

    Returns:
    None
    '''
    dqn_target.load_state_dict(dqn.state_dict())

def _plot(scores: List[float], losses: List[float], epsilons: List[float]) -> None:
    '''
    Plots the training scores, losses, and epsilon values.

    Args:
    scores (List[float]): The training scores.
    losses (List[float]): The training losses.
    epsilons (List[float]): The epsilon values.

    Returns:
    None
    '''

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f'score: {np.mean(scores[-10:]) if scores else 0:.2f}')
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(133)
    plt.title('epsilons')
    plt.plot(epsilons)
    plt.xlabel("Update step")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def select_action(state: np.ndarray, epsilon: float, dqn: Network, device: torch.device,
    env: gym.Env, is_test: bool) -> Tuple[int, List[Any]]:
    '''
    Selects an action using epsilon-greedy policy.

    Args:
    state (np.ndarray): The current state of the environment.
    epsilon (float): The exploration rate.
    dqn (Network): The DQN network.
    device (torch.device): The device to perform computations on.
    env (gym.Env): The environment.
    is_test (bool): Flag indicating if the agent is in test mode.

    Returns:
    Tuple[int, List[Any]]: The selected action and the transition cache.
    '''

    transition_cache = []
    if epsilon > np.random.random():
        selected_action = env.action_space.sample()
    else:
        dqn.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            selected_action = dqn(state_tensor).argmax().item()
        dqn.train()

    if not is_test:
        transition_cache = [state, selected_action]

    return selected_action, transition_cache

def step_env(env: gym.Env, memory: ReplayBuffer, is_test: bool, transition_cache: List[Any],
    action: int) -> Tuple[np.ndarray, float, bool]:
    '''
    Takes an action in the environment and stores the transition.

    Args:
    env (gym.Env): The environment.
    memory (ReplayBuffer): The replay buffer.
    is_test (bool): Flag indicating if the agent is in test mode.
    transition_cache (List[Any]): The last transition.
    action (int): The action to take.

    Returns:
    Tuple[np.ndarray, float, bool]: The next state, reward, and done flag.
    '''
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if not is_test and transition_cache:
        transition = transition_cache + [reward, next_state, done]
        memory.store(*transition)

    return next_state, reward, done

def update_model(memory: ReplayBuffer, dqn: Network, dqn_target: Network,
    optimizer: optim.Optimizer, gamma: float, device: torch.device, epsilon: float,
    min_epsilon: float, max_epsilon: float, epsilon_decay: float, update_cnt: int,
    target_update: int) -> Tuple[float, float, int]:
    '''
    Updates the DQN model using a batch from the replay buffer.

    Args:
    memory (ReplayBuffer): The replay buffer.
    dqn (Network): The DQN network.
    dqn_target (Network): The target DQN network.
    optimizer (optim.Optimizer): The optimizer.
    gamma (float): The discount factor.
    device (torch.device): The device to perform computations on.
    epsilon (float): The exploration rate.
    min_epsilon (float): The minimum exploration rate.
    max_epsilon (float): The maximum exploration rate.
    epsilon_decay (float): The decay rate for epsilon.
    update_cnt (int): The current update count.
    target_update (int): The frequency of target network updates.

    Returns:
    Tuple[float, float, int]: The loss, new epsilon, and updated count.
    '''
    samples = memory.sample_batch()
    loss = _compute_dqn_loss(samples, dqn, dqn_target, gamma, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_epsilon = max(
        min_epsilon,
        epsilon - (max_epsilon - min_epsilon) * epsilon_decay
    )

    new_update_cnt = update_cnt + 1

    if new_update_cnt % target_update == 0:
        _target_hard_update(dqn, dqn_target)

    return loss.item(), new_epsilon, new_update_cnt


def train_agent(env: gym.Env, memory: ReplayBuffer, epsilon_decay: float,
    max_epsilon: float, min_epsilon: float, target_update: int, gamma: float,
    device: torch.device, dqn: Network, dqn_target: Network, optimizer: optim.Optimizer,
    num_frames: int) -> Tuple[Network, List[float], List[float], List[float]]:
    '''
    Trains the DQN agent, and plots the training scores, losses, and epsilon values.

    Args:
    env (gym.Env): The environment.
    memory (ReplayBuffer): The replay buffer.
    epsilon_decay (float): The decay rate for epsilon.
    max_epsilon (float): The maximum exploration rate.
    min_epsilon (float): The minimum exploration rate.
    target_update (int): The frequency of target network updates.
    gamma (float): The discount factor.
    device (torch.device): The device to perform computations on.
    dqn (Network): The DQN network.
    dqn_target (Network): The target DQN network.
    optimizer (optim.Optimizer): The optimizer.
    num_frames (int): The number of frames to train the agent.

    Returns:
    Tuple[Network, List[float], List[float], List[float]]: The trained DQN network,
    training scores, losses, and epsilon values.
    '''

    epsilon = max_epsilon
    update_cnt = 0
    losses = []
    epsilons = []
    scores = []
    current_episode_score = 0

    current_state, _ = env.reset()

    for _ in range(num_frames):
        action, transition_cache = select_action(current_state, epsilon, dqn, device, env, False)
        next_state, reward, done = step_env(env, memory, False, transition_cache, action)

        current_state = next_state
        current_episode_score += reward

        if done:
            scores.append(current_episode_score)
            current_state, _ = env.reset()
            current_episode_score = 0

        if memory.can_sample():
            loss, epsilon, update_cnt = update_model(memory, dqn, dqn_target, optimizer, gamma, device,
                epsilon, min_epsilon, max_epsilon, epsilon_decay,
                update_cnt, target_update)
            losses.append(loss)
            epsilons.append(epsilon)

    return dqn, scores, losses, epsilons


if __name__ == "__main__":
    seed = 777
    num_frames = 50000
    memory_size = 1000
    batch_size = 16
    target_update = 100
    updates_per_decay_period = 2000
    epsilon_decay = 1 / updates_per_decay_period
    learning_rate = 1e-3
    gamma = 0.99
    max_epsilon = 1.0
    min_epsilon = 0.1
    is_test = False

    np.random.seed(seed)
    seed_torch(seed)

    env = gym.make("CartPole-v1", max_episode_steps=500)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqn = Network(obs_dim, action_dim).to(device)
    dqn_target = Network(obs_dim, action_dim).to(device)
    dqn_target.load_state_dict(dqn.state_dict())
    dqn_target.eval()

    optimizer = optim.Adam(dqn.parameters(), lr= learning_rate)
    memory = ReplayBuffer(obs_dim, memory_size, batch_size)

    final_dqn, final_scores, final_losses, final_epsilons = train_agent(env, memory,
        epsilon_decay, max_epsilon, min_epsilon, target_update, gamma, device,
        dqn, dqn_target, optimizer, num_frames)

    _plot(final_scores, final_losses, final_epsilons)
    env.close()
