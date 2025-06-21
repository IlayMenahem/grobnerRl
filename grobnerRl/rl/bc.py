import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def bc_loss(policy, actions, states):
    action_preds = policy(states)
    action_preds = pad_sequence(action_preds, batch_first=True)
    actions = torch.tensor(actions, dtype=torch.long)
    actions_one_hot = F.one_hot(actions, num_classes=action_preds.size(-1)).float()
    loss = F.cross_entropy(action_preds, actions_one_hot)

    return loss


def train_bc(policy, dataloader, num_epochs, optimizer):
    losses = []
    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(dataloader), desc="Epoch Progress", leave=False)

    for epoch in range(num_epochs):
        epoch_losses = []

        for states, actions in dataloader:
            loss = bc_loss(policy, actions, states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_progressbar.update(1)
            epoch_progressbar.set_postfix(loss=loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(loss=avg_epoch_loss)

    return policy
