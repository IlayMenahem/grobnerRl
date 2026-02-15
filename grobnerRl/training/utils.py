import json
import os
from datetime import datetime
from typing import Any

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


def create_metrics_log_path(
    base_dir: str = "logs", hyperparameters: dict[str, Any] | None = None
) -> str:
    """
    Create a unique JSON file path for logging metrics with hyperparameters.

    Args:
        base_dir: Base directory for log files.
        hyperparameters: Dictionary of hyperparameters for this training run.

    Returns:
        Path to the new metrics log file.
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(base_dir, f"metrics_{timestamp}.json")
    
    initial_data = {
        "hyperparameters": hyperparameters or {},
        "run_timestamp": timestamp,
        "metrics": [],
    }
    with open(file_path, "w") as f:
        json.dump(initial_data, f, indent=2)
    
    return file_path


def log_metrics(metrics: dict[str, Any], file_path: str, iteration: int) -> None:
    """
    Append metrics to a JSON log file.

    Args:
        metrics: Dictionary of metrics to log.
        file_path: Path to the JSON log file.
        iteration: Current training iteration.
    """
    log_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {"hyperparameters": {}, "metrics": []}
    
    if "metrics" not in data:
        data = {"hyperparameters": data.get("hyperparameters", {}), "metrics": []}
    
    data["metrics"].append(log_entry)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Logged metrics to {file_path}")
