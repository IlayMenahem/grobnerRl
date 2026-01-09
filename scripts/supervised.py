import os
from typing import Sequence

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
from grobnerRl.models import (
    Extractor,
    GrobnerPolicy,
    IdealModel,
    MonomialEmbedder,
    PairwiseScorer,
    PolynomialEmbedder,
)
from grobnerRl.types import Action, Observation
from grobnerRl.training.supervised import train_model, evaluate_policy, loss_and_accuracy


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
        batched_ideals = np.zeros(
            (batch_size, max_polys, max_monoms, num_vars), dtype=np.float32
        )
        batched_monomial_masks = np.zeros(
            (batch_size, max_polys, max_monoms), dtype=bool
        )
        batched_poly_masks = np.zeros((batch_size, max_polys), dtype=bool)
        batched_selectables = np.full(
            (batch_size, max_polys, max_polys), -np.inf, dtype=np.float32
        )

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
            "selectables": batched_selectables,
        }

        return (
            batched_obs,
            np.array(batched_actions, dtype=np.int32),
            np.array(loss_mask, dtype=np.float32),
        )

    num_vars = 5
    multiple = 4.55
    num_clauses = int(num_vars * multiple)
    ideal_dist = f"{num_vars}-{num_clauses}_sat3"
    ideal_gen = SAT3IdealGenerator(num_vars, num_clauses)
    data_path = os.path.join("data", f"{ideal_dist}.json")
    checkpoint_dir = os.path.join("models", "checkpoints")

    num_epochs = 100
    batch_size = 128
    dataset_size = 2**15
    early_stopping_patience = 5
    min_delta = 1e-3

    monomials_dim = num_vars + 1
    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 4
    ideal_num_heads = 8

    key = jax.random.key(0)
    key, k_monomial, k_polynomial, k_ideal, k_scorer = jax.random.split(key, 5)

    monomial_embedder = MonomialEmbedder(
        monomials_dim, monoms_embedding_dim, k_monomial
    )
    polynomial_embedder = PolynomialEmbedder(
        input_dim=monoms_embedding_dim,
        hidden_dim=polys_embedding_dim,
        hidden_layers=2,
        output_dim=polys_embedding_dim,
        key=k_polynomial,
    )
    ideal_model = IdealModel(polys_embedding_dim, ideal_num_heads, ideal_depth, k_ideal)
    pairwise_scorer = PairwiseScorer(polys_embedding_dim, polys_embedding_dim, k_scorer)
    extractor_eqx = Extractor(
        monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer
    )
    policy = GrobnerPolicy(extractor_eqx)

    learning_rate = 1e-4
    optimizer = optax.nadam(learning_rate)

    env = BuchbergerEnv(ideal_gen)
    expert_policy = BasicExpert(env)

    if not os.path.exists(data_path):
        generate_expert_data(env, dataset_size, data_path, expert_policy)

    full_ds = JsonDatasource(data_path, "states", "actions")
    indices = np.arange(len(full_ds))
    split = int(0.8 * len(full_ds))
    train_indices = indices[:split]
    val_indices = indices[split:]

    val_datasource = JsonDatasource(data_path, "states", "actions", indices=val_indices)
    train_datasource = JsonDatasource(
        data_path, "states", "actions", indices=train_indices
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
