import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def bc_loss(policy, actions, states):
    action_preds = policy(states)
    action_preds = pad_sequence(action_preds, batch_first=True)
    actions = torch.tensor(actions, dtype=torch.long)
    loss = F.cross_entropy(action_preds, actions)

    return loss


def accuracy_fn(policy, actions, states):
    action_preds = policy(states)
    action_preds = pad_sequence(action_preds, batch_first=True)
    actions = torch.tensor(actions, dtype=torch.long)

    _, predicted = torch.max(action_preds, 1)
    correct = (predicted == actions).sum().item()
    accuracy = correct / len(actions)

    return accuracy


def train_model(policy, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn,
    accuercy_fn, scheduler=None, default_device='cpu', checkpointer=None):
    '''
    Train a policy using behavior cloning.

    Args:
    - policy (torch.nn.Module): The policy model to train.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - val_dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the policy.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - accuercy_fn (callable): Function to compute accuracy of the model.
    - scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
    - default_device (str): Device to run the training on ('cpu', 'cuda', or 'mps').
    - checkpointer (Optional[callable]): Function to save checkpoints during training.

    Returns:
    - policy (torch.nn.Module): The trained policy model.
    - losses_train (list): List of average training losses per epoch.
    - losses_validation (list): List of average validation losses per epoch.
    '''
    torch.set_default_device(default_device)

    losses_train = []
    losses_validation = []

    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(train_dataloader), desc="Epoch Progress", leave=False)

    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(train_dataloader, loss_fn, policy, optimizer, epoch_progressbar)
        losses_train.append(avg_epoch_loss)

        # Validate after each epoch
        avg_val_loss = validate_model(policy, val_dataloader, accuercy_fn)
        losses_validation.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_epoch_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(train_loss=avg_epoch_loss, val_loss=avg_val_loss)

    return policy, losses_train, losses_validation


def train_epoch(dataloader, loss_fn, policy, optimizer, epoch_progressbar):
    '''
    Train the policy for one epoch.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - policy (torch.nn.Module): The policy model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the policy.
    - epoch_progressbar (tqdm): Progress bar for the current epoch.

    Returns:
    - avg_epoch_loss (float): Average loss over the epoch.
    '''
    policy.train()
    epoch_losses = []

    for states, actions in dataloader:
        loss = loss_fn(policy, actions, states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

    return avg_epoch_loss


def validate_model(model, dataloader, loss_fn):
    '''
    Validate the model on a validation dataset.

    Args:
    - model (torch.nn.Module): The model to validate.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - loss_fn (callable): Loss function to compute the loss of a batch.

    Returns:
    - avg_loss (float): Average loss over the validation dataset.
    '''
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for states, actions in dataloader:
            loss = loss_fn(model, actions, states)
            total_loss += loss
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    return avg_loss
