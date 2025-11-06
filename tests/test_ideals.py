import itertools
import random

from grobnerRl.envs.ideals import SAT3IdealGenerator


def test_sat3_generator_produces_boolean_constraints_and_clauses():
	num_vars, num_clauses = 4, 6
	generator = SAT3IdealGenerator(num_vars, num_clauses)

	random.seed(0)
	ideal = next(generator)

	assert len(ideal) == num_vars + num_clauses

	ring = ideal[0].ring
	vars_in_ring = ring.gens
	expected_var_polys = [var * (ring.one - var) for var in vars_in_ring]
	assert ideal[:num_vars] == expected_var_polys


def test_sat3_generator_clause_polynomials_match_expected_factors_with_seed():
	num_vars, num_clauses = 5, 4
	generator = SAT3IdealGenerator(num_vars, num_clauses)
	seed = 1234

	rng = random.Random(seed)
	clause_specs = [
		(rng.sample(range(num_vars), 3), rng.choices([0, 1], k=3))
		for _ in range(num_clauses)
	]

	random.seed(seed)
	ideal = next(generator)

	ring = ideal[0].ring
	vars_in_ring = ring.gens
	produced_clauses = ideal[num_vars:]

	assert len(produced_clauses) == num_clauses

	for clause_poly, (indices, negations) in zip(produced_clauses, clause_specs):
		expected = ring.one
		for idx, neg in zip(indices, negations):
			expected *= ring(neg) - vars_in_ring[idx]
		assert clause_poly == expected


def test_sat3_clause_polynomial_vanishes_on_satisfying_assignments():
	num_vars, num_clauses = 3, 1
	generator = SAT3IdealGenerator(num_vars, num_clauses)
	seed = 99

	rng = random.Random(seed)
	clause_indices = rng.sample(range(num_vars), 3)
	clause_negations = rng.choices([0, 1], k=3)

	random.seed(seed)
	ideal = next(generator)
	clause_poly = ideal[-1]
	ring = clause_poly.ring
	domain = ring.domain

	for assignment in itertools.product([0, 1], repeat=num_vars):
		literal_truths = [
			assignment[idx] if neg else 1 - assignment[idx]
			for idx, neg in zip(clause_indices, clause_negations)
		]
		clause_truth = any(literal_truths)
		value = clause_poly(*assignment)
		if clause_truth:
			assert value == domain.zero
		else:
			assert value == domain.one