import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
from tqdm import tqdm
from implementations.utils import plot_learning_process
from grobnerRl.envs.deepgroebner import BuchbergerEnv, BuchbergerAgent
from grobnerRl.models import Extractor, GrobnerPolicy, GrobnerCritic
from grobnerRl.data import JsonDataset, collate, generate_expert_data


def bc_accuracy_and_loss(model, data, labels, device):
    """Optimized BC loss with explicit device handling."""
    logits_list = model(data)

    # Move labels once
    if labels.device != device:
        labels = labels.to(device, non_blocking=True)

    padded_logits = torch.nn.utils.rnn.pad_sequence(logits_list, batch_first=True)

    loss = F.cross_entropy(padded_logits, labels)

    with torch.no_grad():
        predictions = torch.argmax(padded_logits, dim=-1)
        correct = (predictions == labels).float().sum()
        accuracy = correct / len(logits_list)

    return loss, accuracy


def value_accuracy_and_loss(model, data, returns, device):
    """Optimized value loss with explicit device handling."""
    preds = model(data).view(-1)

    if returns.device != device:
        targets = returns.to(device, non_blocking=True).float().view(-1)
    else:
        targets = returns.float().view(-1)

    loss = F.huber_loss(preds, targets)

    with torch.no_grad():
        accuracy = -F.l1_loss(preds, targets)

    return loss, accuracy


def train_epoch_nested(dataloader, loss_and_accuracy_fn, model, optimizer, epoch_progressbar,
                       device):
    """Optimized training epoch."""
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for data, labels in dataloader:
        # Labels are moved inside loss function for non_blocking transfer

        loss, accuracy = loss_and_accuracy_fn(model, data, labels, device)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
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
    """Optimized validation epoch."""
    model.eval()
    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for data, labels in dataloader:
            loss, accuracy = loss_and_accuracy_fn(model, data, labels, device)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy.item() if torch.is_tensor(accuracy) else accuracy)

    avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_epoch_accuracy = float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0

    return avg_epoch_loss, avg_epoch_accuracy


def train_model_nested(model, train_loader, val_loader, epochs, optimizer, loss_and_accuracy_fn,
                       device='cpu', scheduler=None, early_stopping_patience=None,
                       early_stopping_min_delta=0.0):
    """Optimized training loop."""
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
        train_loss, train_acc = train_epoch_nested(
            train_loader, loss_and_accuracy_fn, model, optimizer,
            epoch_progressbar, device
        )
        losses_train.append(train_loss)
        accuracy_train.append(train_acc)

        # Validation
        val_loss, val_acc = validate_epoch_nested(
            val_loader, loss_and_accuracy_fn, model, device
        )
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


@torch.no_grad()
def evaluate_policy_batch(model, actor_path, env, model_env, device, expert_agent, num_episodes=250):
    """Optimized evaluation with batching where possible."""
    model.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    model.eval()

    model_rewards, expert_rewards = [], []

    # Evaluate model policy
    for episode in tqdm(range(num_episodes), desc='Evaluating model policy'):
        obs, _ = model_env.reset(seed=episode)
        episode_reward, episode_done = 0, False

        while not episode_done:
            logits = model(obs)
            action = int(torch.argmax(logits).item())

            obs, reward, terminated, truncated, _ = model_env.step(action)
            episode_reward += reward
            episode_done = terminated or truncated

        model_rewards.append(episode_reward)

    # Evaluate expert policy
    for episode in tqdm(range(num_episodes), desc='Evaluating expert policy'):
        obs, _ = env.reset(seed=episode)
        episode_reward, episode_done = 0, False

        while not episode_done:
            expert_action = expert_agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(expert_action)
            episode_reward += reward
            episode_done = terminated or truncated

        expert_rewards.append(episode_reward)

    # Print evaluation results
    model_mean_reward = np.mean(model_rewards)
    expert_mean_reward = np.mean(expert_rewards)
    performance_ratio = model_mean_reward / expert_mean_reward if expert_mean_reward != 0 else 0

    print(f"Evaluation complete - Policy reward: {model_mean_reward:.4f}, Expert reward: {expert_mean_reward:.4f}, Performance ratio: {performance_ratio:.4f}")

    return model_rewards, expert_rewards


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    num_vars = 3
    max_degree = 4
    num_polys = 4

    # AUTO-DETECT GPU (CUDA or MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    ideal_dist = f'{num_vars}-{max_degree}-{num_polys}-uniform'
    data_path = f'data/expert_data_{ideal_dist}.json'
    actor_path = 'models/imitation_policy1.pth'
    critic_path = 'models/imitation_critic1.pth'

    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 4
    ideal_num_heads = 4
    extractor_params = (num_vars, monoms_embedding_dim, polys_embedding_dim, ideal_depth, ideal_num_heads)
    lr = 1e-4
    gamma = 0.99

    dataset_size = int(1e6)
    batch_size = 128

    num_workers = 4
    print(f"Using {num_workers} data loading workers")

    epochs = 50

    # Initialize models
    model = GrobnerPolicy(Extractor(*extractor_params))
    critic = GrobnerCritic(Extractor(*extractor_params))
    model.to(device)
    critic.to(device)

    # OPTIMIZATION: Compile model with torch.compile for PyTorch 2.0+ (if available)
    if hasattr(torch, 'compile') and device in ['cuda', 'mps']:
        model = torch.compile(model, mode='default')
        critic = torch.compile(critic, mode='default')


    env = BuchbergerEnv(ideal_dist, mode='train')
    expert_agent = BuchbergerAgent('degree_after_reduce')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    if not os.path.exists(data_path):
        print(f"Generating expert data at {data_path}...")
        generate_expert_data(env, dataset_size, data_path, expert_agent)

    # Load datasets
    print("Loading datasets...")
    dataset = JsonDataset(data_path, 'states', 'actions')
    critic_dataset = JsonDataset(data_path, 'states', 'values')

    split = [0.8, 0.2]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, split)
    critic_train_dataset, critic_val_dataset = torch.utils.data.random_split(critic_dataset, split)

    # OPTIMIZATION: pin_memory for faster GPU transfer, persistent_workers to avoid respawning
    pin_memory = (device in ['cuda', 'mps'])
    persistent_workers = (num_workers > 0)

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size, collate_fn=collate,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    critic_train_loader = DataLoader(
        critic_train_dataset, batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    critic_val_loader = DataLoader(
        critic_val_dataset, batch_size, collate_fn=collate,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # OPTIMIZATION: Use AdamW instead of Adam (generally better)
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    critic_optimizer = optim.AdamW(critic.parameters(), lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, patience=5, factor=0.5)

    # Train policy
    print("\n" + "="*80)
    print("TRAINING POLICY")
    print("="*80)
    model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model_nested(
        model, train_loader, val_loader, epochs, optimizer, bc_accuracy_and_loss,
        device=device, scheduler=scheduler, early_stopping_patience=7,
        early_stopping_min_delta=0.001
    )

    plot_learning_process(losses_train, losses_validation, accuracy_train, accuracy_validation)
    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    torch.save(model.state_dict(), actor_path)
    print(f"Policy saved to {actor_path}")

    # Train critic
    print("\n" + "="*80)
    print("TRAINING CRITIC")
    print("="*80)
    critic_model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model_nested(
        critic, critic_train_loader, critic_val_loader, epochs, critic_optimizer, value_accuracy_and_loss,
        device=device, scheduler=critic_scheduler, early_stopping_patience=7,
        early_stopping_min_delta=0.001
    )

    plot_learning_process(losses_train, losses_validation, accuracy_train, accuracy_validation)
    os.makedirs(os.path.dirname(critic_path), exist_ok=True)
    torch.save(critic_model.state_dict(), critic_path)
    print(f"Critic saved to {critic_path}")

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    env = BuchbergerEnv(ideal_dist)
    model_env = BuchbergerEnv(ideal_dist, mode='train')

    model_rewards, expert_rewards = evaluate_policy_batch(
        model, actor_path, env, model_env, device, expert_agent, num_episodes=250
    )
