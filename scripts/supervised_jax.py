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
import optax
import equinox as eqx
from jaxtyping import Array
from equinox import Module
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.transforms import Batch


def train_model(policy: Module, dataloader_train: DataLoader, dataloader_validation: DataLoader, num_epochs: int, optimizer: optax.GradientTransformation, loss_and_accuracy: Callable)->tuple[Module, list[float], list[float], list[float], list[float]]:
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
    - Training losses (list of float).
    - Training accuracies (list of float).
    - Validation losses (list of float).
    - Validation accuracies (list of float).   
    """
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    def make_step(model, opt_state, observations, actions):
        def loss_fn(m):
            loss, acc = loss_and_accuracy(m, observations, actions)
            return loss, acc

        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    @eqx.filter_jit
    def eval_step(model, observations, actions):
        loss, acc = loss_and_accuracy(model, observations, actions)
        return loss, acc

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_train_acc = []
        
        for observations, actions in dataloader_train:
            policy, opt_state, loss, acc = make_step(policy, opt_state, observations, actions)
            epoch_train_loss.append(loss)
            epoch_train_acc.append(acc)

            print(f"Batch Train Loss: {loss:.4f}, Batch Train Acc: {acc:.4f}", epoch)
        
        t_loss = float(jnp.mean(jnp.array(epoch_train_loss))) if epoch_train_loss else 0.0
        t_acc = float(jnp.mean(jnp.array(epoch_train_acc))) if epoch_train_acc else 0.0
        train_losses.append(t_loss)
        train_accuracies.append(t_acc)

        epoch_val_loss = []
        epoch_val_acc = []
        
        for observations, actions in dataloader_validation:
            loss, acc = eval_step(policy, observations, actions)
            epoch_val_loss.append(loss)
            epoch_val_acc.append(acc)

            print(f"Batch Val Loss: {loss:.4f}, Batch Val Acc: {acc:.4f}")
            
        v_loss = float(jnp.mean(jnp.array(epoch_val_loss))) if epoch_val_loss else 0.0
        v_acc = float(jnp.mean(jnp.array(epoch_val_acc))) if epoch_val_acc else 0.0
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f}, Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")

    return policy, train_losses, train_accuracies, val_losses, val_accuracies

def loss_and_accuracy(model: Module, observations: Sequence[Observation], actions: Sequence[Action]) -> tuple[Array, Array]:
    """
    Compute the loss and accuracy for the given model on the provided observations and actions.

    Args:
    - model (Module): The GrobnerPolicy model.
    - observations (Sequence[Observation]): A batch of observations.
    - actions (Sequence[Action]): A batch of actions.

    Returns:
    - loss (Array): The computed loss.
    - accuracy (Array): The computed accuracy.
    """
    logits_per_obs = [model(obs) for obs in observations]
    max_logit_len = max(logit.shape[0] for logit in logits_per_obs)

    def pad_to_max_length(array, target_len):
        pad_width = [(0, target_len - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
        return jnp.pad(array, pad_width, constant_values=-jnp.inf)

    logits = jnp.stack([pad_to_max_length(logit, max_logit_len) for logit in logits_per_obs])
    targets = jnp.array(actions)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    predicted_actions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predicted_actions == targets)

    return loss, accuracy

if __name__ == "__main__":
    def batch_fn(
        x: Sequence[tuple[Observation, Action]],
    ) -> tuple[Sequence[Observation], Sequence[Action]]:
        observations, actions = zip(*x)

        return observations, actions

    num_vars = 3
    multiple = 4.55
    num_clauses = int(num_vars * multiple)
    ideal_dist = f"{num_vars}-{num_clauses}_sat3"
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    data_path = os.path.join("data", f"{ideal_dist}.json")
    actor_path = os.path.join("models", "imitation_policy.pth")
    critic_path = os.path.join("models", "imitation_critic.pth")
    device = "cpu"
    
    num_epochs = 10
    batch_size = 64
    dataset_size = 1024

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
    train_sampler = IndexSampler(len(datasource), ShardOptions(0, 1, True), True, seed=0)
    train_dataloader = DataLoader(
        data_source=datasource, sampler=train_sampler, operations=(to_batch,), worker_count=1
    )
    val_sampler = IndexSampler(len(datasource), ShardOptions(0, 1, True), True, seed=1)
    val_dataloader = DataLoader(
        data_source=datasource, sampler=val_sampler, operations=(to_batch,), worker_count=1
    )

    model, losses_train, accuracy_train, losses_validation, accuracy_validation = train_model(policy, train_dataloader, val_dataloader, num_epochs, optimizer, loss_and_accuracy) 