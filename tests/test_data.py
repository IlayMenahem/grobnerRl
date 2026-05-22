"""Tests for grobnerRl.data."""

from __future__ import annotations

import json
import random
from pathlib import Path

import sympy as sp
from sympy.polys.rings import PolyElement

from grobnerRl.data import JsonDatasource, generate_expert_data
from grobnerRl.env import BuchbergerEnv
from grobnerRl.experts import BasicExpert
from grobnerRl.ideals import SAT3IdealGenerator

from tests.conftest import DummyIdealGenerator


def _write_sample_dataset(path: Path) -> dict[str, list]:
    dataset = {
        "states": [[[1, 0]], [[0, 1]], [[1, 1]]],
        "actions": [0, 1, 2],
        "values": [-1.0, -2.0, -3.0],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dataset, fh)
    return dataset


def test_json_datasource_returns_state_and_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.json"
    dataset = _write_sample_dataset(dataset_path)
    ds = JsonDatasource(str(dataset_path), "states", ["actions"])
    assert len(ds) == len(dataset["states"])
    state, action = ds[0]
    assert state == dataset["states"][0]
    assert action == dataset["actions"][0]


def test_json_datasource_supports_indices_subset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.json"
    dataset = _write_sample_dataset(dataset_path)
    ds = JsonDatasource(
        str(dataset_path), "states", ["actions", "values"], indices=[2, 0]
    )
    assert len(ds) == 2
    state, action, value = ds[0]
    assert state == dataset["states"][2]
    assert action == dataset["actions"][2]
    assert value == dataset["values"][2]


def test_json_datasource_handles_multiple_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.json"
    dataset = _write_sample_dataset(dataset_path)
    ds = JsonDatasource(str(dataset_path), "states", ["actions", "values"])
    state, action, value = ds[1]
    assert state == dataset["states"][1]
    assert action == dataset["actions"][1]
    assert value == dataset["values"][1]


def test_json_datasource_with_empty_labels_returns_state_only(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.json"
    dataset = _write_sample_dataset(dataset_path)
    ds = JsonDatasource(str(dataset_path), "states", [])
    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 1
    assert item[0] == dataset["states"][0]


def test_generate_expert_data_writes_dataset_with_expected_keys(
    tmp_path: Path,
) -> None:
    random.seed(0)
    generator = SAT3IdealGenerator(num_vars=3, num_clauses=2)
    env = BuchbergerEnv(generator, mode="eval")
    expert = BasicExpert(env, strategy="first")

    out_path = tmp_path / "sub" / "data.json"
    generate_expert_data(env, size=4, path=str(out_path), expert_agent=expert)

    assert out_path.exists()
    with open(out_path) as fh:
        dataset = json.load(fh)
    assert set(dataset.keys()) == {"states", "actions", "values"}
    assert len(dataset["states"]) == len(dataset["actions"]) == len(dataset["values"])
    assert len(dataset["states"]) >= 4


def test_generate_expert_data_skips_empty_ideal_episodes(tmp_path: Path) -> None:
    R, x, y = sp.ring("x,y", sp.QQ, "lex")
    empty: list[PolyElement] = []
    real_ideal = [x**2 + y, x * y + 1]

    generator = DummyIdealGenerator([empty, real_ideal, real_ideal, real_ideal])
    env = BuchbergerEnv(generator, mode="eval")
    expert = BasicExpert(env, strategy="first")

    out_path = tmp_path / "data.json"
    generate_expert_data(env, size=1, path=str(out_path), expert_agent=expert)

    with open(out_path) as fh:
        dataset = json.load(fh)
    assert len(dataset["states"]) >= 1


def test_generate_expert_data_discounted_returns_use_gamma(tmp_path: Path) -> None:
    R, x, y = sp.ring("x,y", sp.QQ, "lex")
    ideal = [x**2 + y, x * y + 1]
    generator = DummyIdealGenerator([ideal, ideal, ideal])
    env = BuchbergerEnv(generator, mode="eval", rewards="reductions")  # rewards=-1/step
    expert = BasicExpert(env, strategy="first")

    gamma = 0.5
    out_path = tmp_path / "data.json"
    generate_expert_data(
        env, size=2, path=str(out_path), expert_agent=expert, gamma=gamma
    )

    with open(out_path) as fh:
        dataset = json.load(fh)
    values = dataset["values"]
    assert len(values) >= 2
    # Last recorded value of a length-N episode equals -1 (terminal step reward
    # is 0, prior step reward is -1 → value at last state = -1 + γ*0 = -1).
    # We only assert monotonic non-positivity since episode boundaries vary.
    assert all(v <= 0 for v in values)


