import os

import jax
import numpy as np
import optax
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.transforms import Batch

from grobnerRl.data import JsonDatasource, generate_expert_data
from grobnerRl.envs.env import BuchbergerEnv
from grobnerRl.envs.ideals import SAT3IdealGenerator
from grobnerRl.experts import BasicExpert
from grobnerRl.models import GrobnerPolicyValue, ModelConfig
from grobnerRl.training.supervised import (
    batch_fn,
    evaluate_policy,
    loss_and_accuracy,
    train_model,
)

if __name__ == "__main__":
    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)
    ideal_dist = f"{num_vars}-{num_clauses}_sat3"
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    data_path = os.path.join("data", f"{ideal_dist}.json")
    checkpoint_dir = os.path.join("models", f"checkpoints")

    num_epochs = 100
    batch_size = 128
    dataset_size = 2**15
    early_stopping_patience = 5
    min_delta = 1e-3

    monomials_dim = num_vars + 1
    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 2
    ideal_num_heads = 2
    value_hidden_dim = 128

    config = ModelConfig(
        monomials_dim=monomials_dim,
        monoms_embedding_dim=monoms_embedding_dim,
        polys_embedding_dim=polys_embedding_dim,
        ideal_depth=ideal_depth,
        ideal_num_heads=ideal_num_heads,
        value_hidden_dim=value_hidden_dim,
    )

    key = jax.random.key(0)
    policy = GrobnerPolicyValue.from_scratch(config, key)

    learning_rate = 1e-4
    optimizer = optax.nadam(learning_rate)

    env = BuchbergerEnv(ideal_gen)
    expert_policy = BasicExpert(env)

    if not os.path.exists(data_path):
        generate_expert_data(env, dataset_size, data_path, expert_policy)

    full_ds = JsonDatasource(data_path, "states", ["actions", "values"])
    indices = np.arange(len(full_ds))
    split = int(0.8 * len(full_ds))
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()

    val_datasource = JsonDatasource(
        data_path, "states", ["actions", "values"], indices=val_indices
    )
    train_datasource = JsonDatasource(
        data_path, "states", ["actions", "values"], indices=train_indices
    )

    to_batch = Batch(batch_size, True, batch_fn)

    train_sampler = IndexSampler(
        len(train_datasource), ShardOptions(0, 1, True), True, 1, seed=0
    )
    train_dataloader = DataLoader(
        data_source=train_datasource,
        sampler=train_sampler,
        operations=(to_batch,),
        worker_count=1,
        worker_buffer_size=4,
    )
    val_sampler = IndexSampler(
        len(val_datasource), ShardOptions(0, 1, True), True, 1, seed=1
    )
    val_dataloader = DataLoader(
        data_source=val_datasource,
        sampler=val_sampler,
        operations=(to_batch,),
        worker_count=1,
        worker_buffer_size=4,
    )

    model, losses_train, accuracy_train, losses_validation, accuracy_validation = (
        train_model(
            policy,
            train_dataloader,
            val_dataloader,
            num_epochs,
            optimizer,
            loss_and_accuracy,
            checkpoint_dir,
            early_stopping_patience,
            min_delta,
        )
    )

    eval_policy_env = BuchbergerEnv(
        SAT3IdealGenerator(num_vars, num_clauses), mode="train"
    )
    eval_expert_env = BuchbergerEnv(SAT3IdealGenerator(num_vars, num_clauses))
    eval_expert = BasicExpert(eval_expert_env)
    episodes = 100

    evaluate_policy(model, eval_policy_env, eval_expert_env, eval_expert, episodes)
