import os

import equinox as eqx
import optax
from equinox import Module


def load_checkpoint(checkpoint_path: str, template: dict) -> dict:
    """
    Load a checkpoint saved by supervised_jax.py save_checkpoint().

    Args:
        checkpoint_path: Path to the .eqx checkpoint file.
        template: A dictionary with the same structure as the saved payload,
                  containing model templates for deserialization.

    Returns:
        Dictionary containing: model, opt_state, epoch, val_accuracy
    """
    with open(checkpoint_path, "rb") as f:
        payload = eqx.tree_deserialise_leaves(f, template)
    return payload


def save_checkpoint(
    model: Module,
    opt_state: optax.OptState,
    checkpoint_dir: str,
    label: str,
    iteration: int,
    metrics: dict,
) -> str:
    """
    Save a checkpoint during AlphaZero training.

    Args:
        model: The GrobnerAlphaZero model to save.
        opt_state: Current optimizer state.
        checkpoint_dir: Directory to save checkpoints.
        label: Label for the checkpoint (e.g., 'last', 'best').
        iteration: Current training iteration.
        metrics: Dictionary of training metrics.

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{label}.eqx")
    payload = {
        "model": model,
        "opt_state": opt_state,
        "iteration": iteration,
        "metrics": metrics,
    }
    with open(ckpt_path, "wb") as f:
        eqx.tree_serialise_leaves(f, payload)
    return ckpt_path
