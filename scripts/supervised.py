import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sympy

import os
import numpy as np
from tqdm import tqdm
from implementations.utils import plot_learning_process
from grobnerRl.envs.deepgroebner import BuchbergerEnv, BuchbergerAgent
from grobnerRl.envs.ideals import RandomIdealGenerator
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic
from grobnerRl.data import JsonDataset, collate, generate_expert_data


def bc_accuracy_and_loss(model, data, labels):
    logits_list = model(data)

    device = logits_list[0].device
    labels = labels.to(device)
    padded_logits = torch.nn.utils.rnn.pad_sequence(logits_list, batch_first=True).to(device)

    loss = F.cross_entropy(padded_logits, labels)

    with torch.no_grad():
        predictions = torch.argmax(padded_logits, dim=-1)
        correct = (predictions == labels).float().sum()
        accuracy = correct / len(logits_list)

    return loss, accuracy


def value_accuracy_and_loss(model, data, returns):
    preds = model(data).view(-1)
    targets = returns.to(preds.device).float().view(-1)

    loss = F.huber_loss(preds, targets)

    with torch.no_grad():
        accuracy = -F.l1_loss(preds, targets)

    return loss, accuracy


def train_epoch_nested(dataloader, loss_and_accuracy_fn, model, optimizer, epoch_progressbar, device):
    """Custom training epoch that handles nested tensor batches."""
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for data, labels in dataloader:
        # data is a tuple of (nested_batch, selectables_batch)
        # We don't move data to device here since nested tensors need special handling
        labels = labels.to(device, non_blocking=True)

        loss, accuracy = loss_and_accuracy_fn(model, data, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy.item() if torch.is_tensor(accuracy) else accuracy)
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_epoch_accuracy = float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0

    return avg_epoch_loss, avg_epoch_accuracy


def validate_epoch_nested(dataloader, loss_and_accuracy_fn, model, device):
    """Custom validation epoch that handles nested tensor batches."""
    model.eval()
    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for data, labels in dataloader:
            labels = labels.to(device, non_blocking=True)
            loss, accuracy = loss_and_accuracy_fn(model, data, labels)
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy.item() if torch.is_tensor(accuracy) else accuracy)

    avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_epoch_accuracy = float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0

    return avg_epoch_loss, avg_epoch_accuracy


def train_model_nested(model, train_loader, val_loader, epochs, optimizer, loss_and_accuracy_fn,
                       device='cpu', scheduler=None, early_stopping_patience=None, early_stopping_min_delta=0.0):
    """Custom training loop that handles nested tensor batches."""
    losses_train = []
    accuracy_train = []
    losses_validation = []
    accuracy_validation = []

    best_val_loss = float('inf')
    patience_counter = 0

    epoch_progressbar = tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False)

    for epoch in range(epochs):
        epoch_progressbar.reset()
        epoch_progressbar.set_description(f'Epoch {epoch+1}/{epochs}')

        # Training
        train_loss, train_acc = train_epoch_nested(train_loader, loss_and_accuracy_fn, model,
                                                   optimizer, epoch_progressbar, device)
        losses_train.append(train_loss)
        accuracy_train.append(train_acc)

        # Validation
        val_loss, val_acc = validate_epoch_nested(val_loader, loss_and_accuracy_fn, model, device)
        losses_validation.append(val_loss)
        accuracy_validation.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping
        if early_stopping_patience is not None:
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break

    epoch_progressbar.close()

    return model, losses_train, accuracy_train, losses_validation, accuracy_validation


class ModelAdapter(nn.Module):
    '''
    Adapter class to allow the models defined in grobnerRl.models to be used with BuchbergerEnv.
    '''

    def __init__(self, model, device: str | torch.device | None = None):
        super(ModelAdapter, self).__init__()
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    def forward(self, obs):
        prepared_obs = self._prepare_observation(obs)
        output = self.model(prepared_obs)

        if isinstance(output, list) and len(output) == 1:
            return output[0]

        return output

    def _prepare_observation(self, obs):
        if not isinstance(obs, tuple) or len(obs) != 2:
            raise ValueError("Observation must be a tuple of (ideal, selectables).")

        nested_batch, selectables_batch = obs

        if self._is_prepared(nested_batch):
            nested_batch = [tensor.to(self.device) for tensor in nested_batch]
            return (nested_batch, selectables_batch)

        if isinstance(nested_batch, list):
            return self._prepare_single_state(obs)

        raise ValueError("Unsupported observation format for ModelAdapter.")

    def _prepare_single_state(self, state):
        ideal, selectables = state

        poly_tensors = []
        for poly in ideal:
            if torch.is_tensor(poly):
                poly_tensor = poly.to(self.device)
            else:
                poly_array = np.array(poly, dtype=np.float32)
                poly_tensor = torch.as_tensor(poly_array, dtype=torch.float32, device=self.device)
            poly_tensors.append(poly_tensor)

        nested_ideal = torch.nested.nested_tensor(poly_tensors, layout=torch.jagged).to(self.device)
        selectables_list = [tuple(int(idx) for idx in pair) for pair in selectables]

        return ([nested_ideal], [selectables_list])

    @staticmethod
    def _is_prepared(nested_batch):
        return (
            isinstance(nested_batch, list)
            and len(nested_batch) > 0
            and torch.is_tensor(nested_batch[0])
        )


def evaluate_policy(model, actor_path, env, model_env, device, expert_agent):
    """Evaluate trained policy against expert agent performance."""

    model.load_state_dict(torch.load(actor_path, map_location=device))
    model.eval()
    policy_adapter = ModelAdapter(model, device=device)
    policy_adapter.eval()

    model_rewards, expert_rewards = [], []

    for episode in tqdm(range(250), desc='Evaluating policy'):
        obs, _ = model_env.reset(seed=episode)
        episode_reward, episode_done = 0, False

        while not episode_done:
            with torch.no_grad():
                logits = policy_adapter(obs)
                action = int(torch.argmax(logits).item())

            obs, reward, terminated, truncated, _ = model_env.step(action)
            episode_reward += reward
            episode_done = terminated or truncated

        model_rewards.append(episode_reward)

        obs, _ = env.reset(seed=episode)
        episode_reward, episode_done = 0, False

        while not episode_done:
            # In 'eval' mode, obs is the raw state (G, P), not tokenized
            expert_action = expert_agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(expert_action)
            episode_reward += reward
            episode_done = terminated or truncated

        expert_rewards.append(episode_reward)

    # Print evaluation results
    model_mean_reward = torch.tensor(model_rewards).mean().item()
    expert_mean_reward = torch.tensor(expert_rewards).mean().item()
    performance_ratio = model_mean_reward / expert_mean_reward

    print(f"Evaluation complete - Policy reward: {model_mean_reward:.4f}, Expert reward: {expert_mean_reward:.4f}, Performance ratio: {performance_ratio:.4f}")

    return model_rewards, expert_rewards


if __name__ == "__main__":
    torch.manual_seed(42)
    num_vars = 3
    max_degree = 4
    num_polys = 4
    coeff_field = 37
    length_lambda = 0.5
    ideal_dist = f'{num_vars}-{max_degree}-{num_polys}'
    ideal_gen = RandomIdealGenerator(num_vars, max_degree, num_polys, length_lambda, constants=True, coefficient_ring=sympy.FiniteField(coeff_field))
    device = 'cpu'
    data_path = os.path.join('data', f'{ideal_dist}.json')
    actor_path = os.path.join('models', 'imitation_policy.pth')
    critic_path = os.path.join('models', 'imitation_critic.pth')

    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 4
    ideal_num_heads = 4
    extractor_params = (num_vars, monoms_embedding_dim, polys_embedding_dim, ideal_depth, ideal_num_heads)
    lr = 1e-4
    gamma = 0.99

    dataset_size = int(1e6)
    batch_size = 32
    num_workers = 2
    epochs = 250

    model = GrobnerPolicy(Extractor(*extractor_params))
    critic = GrobnerCritic(Extractor(*extractor_params))
    model.to(device)
    critic.to(device)

    env = BuchbergerEnv(ideal_gen, mode='train')
    expert_agent = BuchbergerAgent('degree_after_reduce')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    if not os.path.exists(data_path):
        generate_expert_data(env, dataset_size, data_path, expert_agent)

    dataset = JsonDataset(data_path, 'states', 'actions')
    critic_dataset = JsonDataset(data_path, 'states', 'values')
    split = [0.9, 0.1]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, split)
    critic_train_dataset, critic_val_dataset = torch.utils.data.random_split(critic_dataset, split)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate, num_workers=num_workers)
    critic_train_loader = DataLoader(critic_train_dataset, batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    critic_val_loader = DataLoader(critic_val_dataset, batch_size, collate_fn=collate, num_workers=num_workers)

    optimizer = optim.Adam(model.parameters(), lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, patience=5, factor=0.5)

    model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model_nested(model, train_loader, val_loader, epochs, optimizer, bc_accuracy_and_loss, device=device, scheduler=scheduler, early_stopping_patience=50, early_stopping_min_delta=0.001)
    plot_learning_process(losses_train, losses_validation, accuracy_train, accuracy_validation)
    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    torch.save(model.state_dict(), actor_path)

    env = BuchbergerEnv(ideal_gen)
    expert_agent = BuchbergerAgent('degree_after_reduce')    
    model_env = BuchbergerEnv(ideal_gen, mode='train')
    model_rewards, expert_rewards = evaluate_policy(model, actor_path, env, model_env, device, expert_agent)

    critic_model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model_nested(critic, critic_train_loader, critic_val_loader, epochs, critic_optimizer, value_accuracy_and_loss, device=device, scheduler=critic_scheduler, early_stopping_patience=50, early_stopping_min_delta=0.001)
    plot_learning_process(losses_train, losses_validation, accuracy_train, accuracy_validation)
    os.makedirs(os.path.dirname(critic_path), exist_ok=True)
    torch.save(critic_model.state_dict(), critic_path)
