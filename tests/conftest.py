"""Shared pytest fixtures and helpers for the test suite."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

import pytest
import sympy as sp
from sympy.polys.rings import PolyElement

from grobnerRl.env import (
    BuchbergerEnv,
    interreduce,
    minimalize,
    spoly,
)
from grobnerRl.env import reduce as poly_reduce
from grobnerRl.ideals import IdealGenerator


class DummyIdealGenerator(IdealGenerator):
    """IdealGenerator that yields a fixed sequence of pre-built ideals."""

    def __init__(self, batches: Iterable[list[PolyElement]]):
        super().__init__()
        self._batches = list(batches)
        self._iter = iter(self._batches)

    def __iter__(self) -> "DummyIdealGenerator":
        return self

    def __next__(self) -> list[PolyElement]:
        return next(self._iter)


def assert_reduced_groebner_basis(basis: list[PolyElement]) -> None:
    """Assert that ``basis`` is monic, interreduced, and Groebner."""
    if not basis:
        return
    assert all(poly.LC == 1 for poly in basis)
    ring = basis[0].ring
    ordered = sorted([poly.copy() for poly in basis], key=lambda f: ring.order(f.LM))
    reduced = interreduce([poly.copy() for poly in ordered])
    assert len(ordered) == len(reduced)
    for original, reduced_poly in zip(ordered, reduced):
        assert original == reduced_poly

    for p, q in itertools.combinations(basis, 2):
        remainder, _ = poly_reduce(spoly(p, q), basis)
        assert remainder == 0


def assert_groebner_basis(generators: list[PolyElement]) -> None:
    """Assert that ``generators`` produce a Groebner basis after reduction."""
    reduced = interreduce(minimalize(generators)) if generators else []
    assert_reduced_groebner_basis(reduced)


@pytest.fixture
def ring_xy_qq_lex() -> tuple[Any, PolyElement, PolyElement]:
    """Return ``(R, x, y)`` over ``sp.QQ`` with lex order."""
    R, x, y = sp.ring("x,y", sp.QQ, "lex")
    return R, x, y


@pytest.fixture
def simple_ideal(ring_xy_qq_lex) -> list[PolyElement]:
    """A small ideal that exercises a few reduction steps."""
    _, x, y = ring_xy_qq_lex
    return [x**2 + y, x * y + 1]


@pytest.fixture
def dummy_ideal_generator(simple_ideal) -> DummyIdealGenerator:
    """A DummyIdealGenerator pre-loaded with the simple ideal."""
    return DummyIdealGenerator([simple_ideal])


@pytest.fixture
def simple_buchberger_env(simple_ideal) -> BuchbergerEnv:
    """BuchbergerEnv wrapping the simple ideal in eval mode."""
    env = BuchbergerEnv(DummyIdealGenerator([simple_ideal]), mode="eval")
    env.reset()
    return env
