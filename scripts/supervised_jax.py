import os
from typing import Sequence, Callable

from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.types import Observation, Action
from grobnerRl.data import JsonDatasource, generate_expert_data
from grobnerRl.models import MonomialEmbedder, PolynomialEmbedder, IdealModel, GrobnerPolicy, Extractor, PairwiseScorer
from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.experts import BasicExpert

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from jaxtyping import Array
from equinox import Module
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.transforms import Batch


def train_model(policy: Module, dataloader_train: DataLoader, dataloader_validation: DataLoader, num_epochs: int, optimizer: optax.GradientTransformation, loss_and_accuracy: Callable) -> tuple[Module, Array, Array, Array, Array]:
    """
    Train the model using supervised learning.

    Args:
    - policy (Module): The GrobnerPolicy model to be trained.
    - dataloader_train (DataLoader): DataLoader for training data.
    - dataloader_validation (DataLoader): DataLoader for validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (optax.GradientTransformation): Optax optimizer.
    - loss_and_accuracy (Callable): Function to compute loss and accuracy.
    
    Returns:
    - Trained model (Module).
    - Training losses (Array).
    - Training accuracies (Array).
    - Validation losses (Array).
    - Validation accuracies (Array).   
    """
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    @eqx.filter_jit
    def make_step(model: Module, opt_state: optax.OptState, observations: dict, actions: Array, loss_mask: Array) -> tuple[Module, optax.OptState, Array, Array]:
        def loss_fn(m):
            loss, acc = loss_and_accuracy(m, observations, actions, loss_mask)
            return loss, acc

        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    @eqx.filter_jit
    def eval_step(model: Module, observations: dict, actions: Array, loss_mask: Array) -> tuple[Array, Array]:
        loss, acc = loss_and_accuracy(model, observations, actions, loss_mask)
        return loss, acc

    @eqx.filter_jit
    def train_epoch(policy: Module, opt_state: optax.OptState) -> tuple[Module, optax.OptState, Array, Array]:
        epoch_losses = []
        epoch_accs = []
        
        for observations, actions, loss_mask in dataloader_train:
            policy, opt_state, loss, acc = make_step(policy, opt_state, observations, actions, loss_mask)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        
        loss = jnp.mean(jnp.array(epoch_losses))
        accuracy = jnp.mean(jnp.array(epoch_accs))

        return policy, opt_state, loss, accuracy

    @eqx.filter_jit
    def validate_epoch(policy: Module) -> tuple[Array, Array]:
        epoch_losses = []
        epoch_accs = []
        
        for observations, actions, loss_mask in dataloader_validation:
            loss, acc = eval_step(policy, observations, actions, loss_mask)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        
        loss = jnp.mean(jnp.array(epoch_losses))
        accuracy = jnp.mean(jnp.array(epoch_accs))

        return loss, accuracy

    train_losses: list[Array] = []
    train_accuracies: list[Array] = []
    val_losses: list[Array] = []
    val_accuracies: list[Array] = []

    for epoch in range(num_epochs):
        policy, opt_state, t_loss, t_acc = train_epoch(policy, opt_state)
        v_loss, v_acc = validate_epoch(policy)

        train_losses.append(t_loss)
        train_accuracies.append(t_acc)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {float(t_loss):.4f}, Train Acc: {float(t_acc):.4f}, "
            f"Val Loss: {float(v_loss):.4f}, Val Acc: {float(v_acc):.4f}"
        )

    return (
        policy,
        jnp.stack(train_losses),
        jnp.stack(train_accuracies),
        jnp.stack(val_losses),
        jnp.stack(val_accuracies),
    )

def loss_and_accuracy(model: Module, observations: dict, actions: Array, loss_mask: Array) -> tuple[Array, Array]:
    """
    Compute the loss and accuracy for the given model on the provided observations and actions.

    Args:
    - model (Module): The GrobnerPolicy model.
    - observations (dict): A batch of observations (padded).
    - actions (Array): A batch of actions.
    - loss_mask (Array): A batch of loss masks (1.0 for valid, 0.0 for invalid).

    Returns:
    - loss (Array): The computed loss.
    - accuracy (Array): The computed accuracy.
    """
    logits = eqx.filter_vmap(model)(observations)

    per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions)
    
    # Apply mask
    loss = (per_sample_loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)

    predicted_actions = jnp.argmax(logits, axis=-1)
    correct = (predicted_actions == actions) * loss_mask
    accuracy = correct.sum() / (loss_mask.sum() + 1e-9)

    return loss, accuracy

if __name__ == "__main__":
    def batch_fn(
        x: Sequence[tuple[Observation, Action]],
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        observations, actions = zip(*x)
        batch_size = len(observations)
        
        # 1. Calculate dimensions
        max_polys = max(len(obs[0]) for obs in observations)
        max_monoms = max(max(len(p) for p in obs[0]) for obs in observations)
        num_vars = len(observations[0][0][0][0])

        # 2. Allocate buffers
        batched_ideals = np.zeros((batch_size, max_polys, max_monoms, num_vars), dtype=np.float32)
        batched_monomial_masks = np.zeros((batch_size, max_polys, max_monoms), dtype=bool)
        batched_poly_masks = np.zeros((batch_size, max_polys), dtype=bool)
        batched_selectables = np.full((batch_size, max_polys, max_polys), -np.inf, dtype=np.float32)
        
        batched_actions = []
        loss_mask = []

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
            "selectables": batched_selectables
        }
        
        return batched_obs, np.array(batched_actions, dtype=np.int32), np.array(loss_mask, dtype=np.float32)

    num_vars = 8
    multiple = 4.55
    num_clauses = int(num_vars * multiple)
    ideal_dist = f"{num_vars}-{num_clauses}_sat3"
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    data_path = os.path.join("data", f"{ideal_dist}.json")
    actor_path = os.path.join("models", "imitation_policy.pth")
    critic_path = os.path.join("models", "imitation_critic.pth")
    device = "cpu"
    
    num_epochs = 10 
    batch_size = 128
    dataset_size = 2**14

    # init models
    monomials_dim = num_vars + 1
    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 4
    ideal_num_heads = 8
 
    # Create JAX random keys
    key = jax.random.key(0)
    key, k_monomial, k_polynomial, k_ideal, k_scorer = jax.random.split(key, 5)

    # Build Equinox modules
    monomial_embedder = MonomialEmbedder(monomials_dim, monoms_embedding_dim, k_monomial)
    polynomial_embedder = PolynomialEmbedder(
        input_dim=monoms_embedding_dim,
        hidden_dim=polys_embedding_dim,
        hidden_layers=2,
        output_dim=polys_embedding_dim,
        key=k_polynomial,
    )
    ideal_model = IdealModel(polys_embedding_dim, ideal_num_heads, ideal_depth, k_ideal)
    pairwise_scorer = PairwiseScorer(polys_embedding_dim, polys_embedding_dim, k_scorer)
    extractor_eqx = Extractor(monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer)
    policy = GrobnerPolicy(extractor_eqx)

    optimizer = optax.nadam(1e-3)

    env = BuchbergerEnv(ideal_gen)
    expert_policy = BasicExpert(env)

    if not os.path.exists(data_path):
        generate_expert_data(env, dataset_size, data_path, expert_policy)
    
    to_batch = Batch(batch_size, True, batch_fn)
    datasource = JsonDatasource(data_path, "states", "actions")
    train_sampler = IndexSampler(len(datasource) , ShardOptions(0, 1, True), True, 1, seed=0)
    train_dataloader = DataLoader(
        data_source=datasource, sampler=train_sampler, operations=(to_batch,), worker_count=4
    )
    val_sampler = IndexSampler(len(datasource), ShardOptions(0, 1, True), True, 1, seed=1)
    val_dataloader = DataLoader(
        data_source=datasource, sampler=val_sampler, operations=(to_batch,), worker_count=4
    )

    model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model(policy, train_dataloader, val_dataloader, num_epochs, optimizer, loss_and_accuracy) 