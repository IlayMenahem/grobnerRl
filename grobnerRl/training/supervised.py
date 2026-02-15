from typing import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module
from grain import DataLoader
from jaxtyping import Array
from tqdm import tqdm

from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.experts import Expert
from grobnerRl.training.utils import save_checkpoint
from grobnerRl.types import Action, Observation


def batch_fn(
    x: Sequence[tuple[Observation, Action, float]],
) -> tuple[dict, Array, Array, Array]:
    observations, actions, values = zip(*x)
    batch_size: int = len(observations)

    # 1. Calculate dimensions
    max_polys: int = max(len(obs[0]) for obs in observations)
    max_monoms: int = max(max(len(p) for p in obs[0]) for obs in observations)
    num_vars = len(observations[0][0][0][0])

    # 2. Allocate buffers
    batched_ideals: np.ndarray = np.zeros(
        (batch_size, max_polys, max_monoms, num_vars), dtype=np.float32
    )
    batched_monomial_masks: np.ndarray = np.zeros(
        (batch_size, max_polys, max_monoms), dtype=bool
    )
    batched_poly_masks: np.ndarray = np.zeros((batch_size, max_polys), dtype=bool)
    batched_selectables: np.ndarray = np.full(
        (batch_size, max_polys, max_polys), -np.inf, dtype=np.float32
    )

    batched_actions: list[int] = []
    loss_mask: list[float] = []

    for i, (ideal, selectables) in enumerate(observations):
        num_polys = len(ideal)
        batched_poly_masks[i, :num_polys] = True

        for j, poly in enumerate(ideal):
            p_len = len(poly)
            batched_ideals[i, j, :p_len] = poly
            batched_monomial_masks[i, j, :p_len] = True

        if selectables:
            rows, cols = zip(*selectables)
            batched_selectables[i, rows, cols] = 0.0

            # Remap action index
            r, c = divmod(actions[i], num_polys)
            batched_actions.append(r * max_polys + c)
            loss_mask.append(1.0)
        else:
            batched_selectables[i, 0, 0] = 0.0
            batched_actions.append(0)
            loss_mask.append(0.0)

    batched_obs = {
        "ideals": batched_ideals,
        "monomial_masks": batched_monomial_masks,
        "poly_masks": batched_poly_masks,
        "selectables": batched_selectables,
    }

    batched_values = jnp.array([float(v) for v in values], dtype=jnp.float32)

    return (
        batched_obs,
        jnp.array(batched_actions, dtype=jnp.int32),
        batched_values,
        jnp.array(loss_mask, dtype=jnp.float32),
    )


def train_model(
    policy: Module,
    dataloader_train: DataLoader,
    dataloader_validation: DataLoader,
    num_epochs: int,
    optimizer: optax.GradientTransformation,
    loss_and_accuracy: Callable,
    checkpoint_dir: str | None = None,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
) -> tuple[Module, Array, Array, Array, Array]:
    """
    Train the model using supervised learning.

    Args:
    - policy (Module): The GrobnerPolicy model to be trained.
    - dataloader_train (DataLoader): DataLoader for training data.
    - dataloader_validation (DataLoader): DataLoader for validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (optax.GradientTransformation): Optax optimizer.
    - loss_and_accuracy (Callable): Function to compute loss and accuracy.
    - checkpoint_dir (str | None): Directory to save checkpoints. If None, no checkpoints are written.
    - early_stopping_patience (int | None): Stop if validation accuracy does not improve for this many epochs. If None, early stopping is disabled.
    - min_delta (float): Minimum improvement in validation accuracy to reset the early-stopping counter.

    Returns:
    - Trained model (Module).
    - Training losses (Array).
    - Training accuracies (Array).
    - Validation losses (Array).
    - Validation accuracies (Array).
    """
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: Module,
        opt_state: optax.OptState,
        observations: dict,
        actions: Array,
        values: Array,
        loss_mask: Array,
    ) -> tuple[Module, optax.OptState, Array, Array]:
        def loss_fn(m):
            loss, acc = loss_and_accuracy(m, observations, actions, values, loss_mask)
            return loss, acc

        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    def eval_step(
        model: Module,
        observations: dict,
        actions: Array,
        values: Array,
        loss_mask: Array,
    ) -> tuple[Array, Array]:
        loss, acc = loss_and_accuracy(model, observations, actions, values, loss_mask)
        return loss, acc

    def train_epoch(
        policy: Module, opt_state: optax.OptState
    ) -> tuple[Module, optax.OptState, Array, Array]:
        epoch_losses = []
        epoch_accs = []

        for observations, actions, values, loss_mask in dataloader_train:
            policy, opt_state, loss, acc = make_step(
                policy, opt_state, observations, actions, values, loss_mask
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        loss = jnp.mean(jnp.array(epoch_losses))
        accuracy = jnp.mean(jnp.array(epoch_accs))

        return policy, opt_state, loss, accuracy

    def validate_epoch(policy: Module) -> tuple[Array, Array]:
        epoch_losses = []
        epoch_accs = []

        for observations, actions, values, loss_mask in dataloader_validation:
            loss, acc = eval_step(policy, observations, actions, values, loss_mask)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        loss = jnp.mean(jnp.array(epoch_losses))
        accuracy = jnp.mean(jnp.array(epoch_accs))

        return loss, accuracy

    train_losses: list[Array] = []
    train_accuracies: list[Array] = []
    val_losses: list[Array] = []
    val_accuracies: list[Array] = []
    best_val_acc = float("-inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        policy, opt_state, t_loss, t_acc = train_epoch(policy, opt_state)
        v_loss, v_acc = validate_epoch(policy)

        train_losses.append(t_loss)
        train_accuracies.append(t_acc)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

        val_acc_value = float(v_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {float(t_loss):.4f}, Train Acc: {float(t_acc):.4f}, "
            f"Val Loss: {float(v_loss):.4f}, Val Acc: {val_acc_value:.4f}"
        )

        improved = val_acc_value > best_val_acc + min_delta
        if improved:
            best_val_acc = val_acc_value
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if checkpoint_dir:
            save_checkpoint(
                policy, opt_state, checkpoint_dir, "last", epoch + 1, val_acc_value
            )

            if improved:
                save_checkpoint(
                    policy, opt_state, checkpoint_dir, "best", epoch + 1, val_acc_value
                )

        if (
            early_stopping_patience is not None
            and no_improve_epochs >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1} (no val accuracy improvement > {min_delta} for {early_stopping_patience} epochs)."
            )
            break

    return (
        policy,
        jnp.stack(train_losses),
        jnp.stack(train_accuracies),
        jnp.stack(val_losses),
        jnp.stack(val_accuracies),
    )


def loss_and_accuracy(
    model: Module, observations: dict, actions: Array, values: Array, loss_mask: Array
) -> tuple[Array, Array]:
    """
    Compute the loss and accuracy for the given model on the provided observations and actions.

    Args:
    - model (Module): The GrobnerPolicyValue model.
    - observations (dict): A batch of observations (padded).
    - actions (Array): A batch of actions.
    - values (Array): A batch of target values.
    - loss_mask (Array): A batch of loss masks (1.0 for valid, 0.0 for invalid).

    Returns:
    - loss (Array): The computed loss (policy + value).
    - accuracy (Array): The computed accuracy.
    """
    policy_logits, pred_values = eqx.filter_vmap(model)(observations)

    per_sample_policy_loss = optax.softmax_cross_entropy_with_integer_labels(
        policy_logits, actions
    )

    per_sample_value_loss = optax.huber_loss(pred_values, values)

    per_sample_loss = per_sample_policy_loss + per_sample_value_loss

    # Apply mask
    loss = (per_sample_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)

    predicted_actions = jnp.argmax(policy_logits, axis=-1)
    correct = (predicted_actions == actions) * loss_mask
    accuracy = correct.sum() / (loss_mask.sum() + 1e-9)

    return loss, accuracy


def evaluate_policy(
    policy: Module,
    policy_env: BuchbergerEnv,
    expert_env: BuchbergerEnv,
    expert_agent: Expert,
    episodes: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out the trained policy and compare it to an expert agent.

    Args:
    - policy (Module): Trained GrobnerPolicy.
    - policy_env (BuchbergerEnv): Environment configured with tokenized observations for the policy.
    - expert_env (BuchbergerEnv): Environment with symbolic observations for the expert.
    - expert_agent (Expert): Heuristic expert policy.
    - episodes (int): Number of evaluation episodes.

    Returns:
    - Tuple of numpy arrays with policy rewards and expert rewards per episode.
    """
    pbar = tqdm(range(episodes), desc="Evaluating Policy")

    model_rewards: list[float] = []
    expert_rewards: list[float] = []

    for episode in pbar:
        obs, _ = policy_env.reset(seed=episode)
        ep_reward = 0.0
        done = False

        while not done:
            if hasattr(policy, "extractor"):
                logits, _ = policy(obs)
            else:
                logits = policy(obs)
            action = int(jnp.argmax(logits))

            obs, reward, terminated, truncated, _ = policy_env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        model_rewards.append(ep_reward)

        obs_exp, _ = expert_env.reset(seed=episode)
        exp_reward = 0.0
        done = False

        while not done:
            expert_action = expert_agent(obs_exp)
            obs_exp, reward, terminated, truncated, _ = expert_env.step(expert_action)
            exp_reward += float(reward)
            done = terminated or truncated

        expert_rewards.append(exp_reward)

    model_mean = float(np.mean(model_rewards))
    expert_mean = float(np.mean(expert_rewards))
    ratio = model_mean / expert_mean

    print(
        f"Evaluation complete - Policy reward: {model_mean:.4f}, "
        f"Expert reward: {expert_mean:.4f}, Performance ratio: {ratio:.4f}"
    )

    return np.array(model_rewards, dtype=np.float32), np.array(
        expert_rewards, dtype=np.float32
    )
