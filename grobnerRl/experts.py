from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from sympy.polys.rings import PolyElement

from grobnerRl.envs.env import BaseEnv, GVW_buchberger, reduce, spoly


def select(
    G: list[PolyElement], P: list[tuple[int, int]], strategy="normal"
) -> tuple[int, int]:
    """
    Select and return a pair from P, using the specified strategy.

    Args:
    - G: List of polynomial generators.
    - P: List of s-pairs.
    - strategy: Selection strategy ('first', 'normal', 'degree', 'random', 'degree_after_reduce', or list of these)

    Returns:
    - Selected pair (i, j) from P.
    """

    if not len(G) > 0:
        raise ValueError("polynomial list must be nonempty")

    if not len(P) > 0:
        raise ValueError("pair set must be nonempty")

    R = G[0].ring

    if isinstance(strategy, str):
        strategy = [strategy]

    def strategy_key(p, s):
        """Return a sort key for pair p in the strategy s."""

        if s == "first":
            return p[1], p[0]
        elif s == "normal":
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return R.order(lcm)
        elif s == "degree":
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return sum(lcm)
        elif s == "random":
            return np.random.rand()
        else:
            raise ValueError("unknown selection strategy")

    return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))


class Expert(ABC):
    """
    Abstract base class for experts.
    """

    env: BaseEnv

    def __init__(self, env: BaseEnv):
        """
        Initialize the expert.

        Args:
        - env (BaseEnv): The current environment representing the Buchberger process.
        """
        self.env = env

    @abstractmethod
    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select a pair to reduce from the given observation.

        Args:
        - observation (tuple[list[PolyElement], list[tuple[int, int]]]): The current observation of the Buchberger process.

        Returns:
        - Selected pair (i, j) to reduce.
        """
        pass

    def update_env(self, new_env: BaseEnv):
        """
        Update the expert's environment.

        Args:
        - new_env (BaseEnv): The new environment to set.
        """
        self.env = new_env


def next_step(
    env: BaseEnv, pair: int | tuple[int, int]
) -> tuple[list[PolyElement], list[tuple[int, int]], BaseEnv]:
    """
    Compute the next step in the Buchberger process using the specified pair.

    Args:
    - env (BaseEnv): The current environment representing the Buchberger process.
    - pair (int | tuple[int, int]): The pair index or pair to be used for the next reduction step.

    Returns:
    - tuple: A tuple containing updated generators (G), pairs (P), and the new environment (new_env) after the step.
    """

    # copy the environment to avoid modifying the original one
    new_env = copy(env)
    new_env.generators = list(env.generators)
    new_env.pairs = list(env.pairs)

    (basis, pairs), _, _, _, _ = new_env.step(pair)

    return basis, pairs, new_env


def get_basis(G: list[PolyElement]) -> list[PolyElement]:
    basis, _ = GVW_buchberger(G)
    return basis


def get_leading_terms(basis: list[PolyElement]) -> set[tuple[int, ...]]:
    return {p.LM for p in basis}


def lm_by_pair(
    env: BaseEnv, G: list[PolyElement], pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], PolyElement]:
    """
    Get the leading monomial of the polynomial that would be yielded by the pair reduction.

    Args:
    - env (BaseEnv): The current environment
    - G: list[PolyElement]: The current generators
    - pairs (list[tuple[int, int]]): The pairs we can reduce

    Returns:
    - dict[tuple[int, int], PolyElement]: A dictionary mapping each pair to its leading monomial.
    """
    leading_monomial_by_pair = {}

    for pair in pairs:
        poly = spoly(G[pair[0]], G[pair[1]])
        remainder, _ = reduce(poly, G)

        if remainder != 0:
            leading_monomial_by_pair[pair] = remainder.LM

    return leading_monomial_by_pair


class BasisBasedExpert(Expert):
    """
    Abstract base class for experts that use the basis of the current generators to select a pair to reduce.
    """

    env: BaseEnv
    basis: list[PolyElement]
    leading_terms: set[tuple[int, ...]]

    def __init__(self, env: BaseEnv):
        """
        Initialize the basis-based expert.

        Args:
        - env (BaseEnv): The current environment representing the Buchberger process.
        """
        self.env = env
        self.basis = get_basis(env.generators)
        self.leading_terms = get_leading_terms(self.basis)

    def need_to_recompute_basis(self) -> bool:
        """
        Check if the basis needs to be recomputed.
        this is done by checking if the current basis fully reduces the current generators.

        Returns:
        - True if the basis needs to be recomputed, False otherwise.
        """

        def fully_reduces(G: list[PolyElement], basis: list[PolyElement]) -> bool:
            """
            Check if the basis fully reduces the generators.

            Args:
            - G: List of generators.
            - basis: List of basis polynomials.

            Returns:
            - True if the basis fully reduces the generators, False otherwise.
            """
            for poly in G:
                remainder, _ = reduce(poly, basis)
                if remainder != 0:
                    return False

            return True

        return not fully_reduces(self.env.generators, self.basis)

    @abstractmethod
    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select a pair to reduce from the given observation.

        Args:
        - observation (tuple[list[PolyElement], list[tuple[int, int]]]): The current observation of the Buchberger process.

        Returns:
        - Selected pair (i, j) to reduce.
        """
        pass


class BasicExpert(Expert):
    """
    Basic expert that selects a pair to reduce using a specified strategy.
    """

    env: BaseEnv
    strategy: str

    def __init__(self, env: BaseEnv, strategy="normal"):
        """
        Initialize the basic expert.

        Args:
        - env (BaseEnv): The current environment representing the Buchberger process.
        - strategy (str): The selection strategy to use.
        """
        super().__init__(env)
        self.env = env
        self.strategy = strategy

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        G, P = observation
        pair = select(G, P, strategy=self.strategy)

        return pair


class LowestLMExpert(Expert):
    """
    Select the pair that would yield the polynomial with lowest order leading monomial.
    """

    env: BaseEnv

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select the pair that would yield the polynomial with lowest order leading monomial.

        Args:
        - observation (tuple[list[PolyElement], list[tuple[int, int]]]): The current observation of the Buchberger process.

        Returns:
        - Selected pair (i, j) to reduce.
        """
        G, P = observation
        order = G[0].ring.order
        leading_monomial_by_pair = lm_by_pair(self.env, G, P)

        return min(
            leading_monomial_by_pair.keys(),
            key=lambda p: order(leading_monomial_by_pair[p]),
            default=P[0],
        )


class LeastRemainingPairsExpert(Expert):
    """
    Select the pair that after being reduced, yields the least remaining pairs.
    """

    env: BaseEnv

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select the pair that after being reduced, yields the least remaining pairs.

        Args:
        - observation (tuple[list[PolyElement], list[tuple[int, int]]]): The current observation of the Buchberger process.
        """
        G, P = observation

        pair_by_num_remaining = {}
        for pair in P:
            G, P, new_env = next_step(self.env, pair)
            pair_by_num_remaining[pair] = len(P)

        return min(pair_by_num_remaining.keys(), key=lambda p: pair_by_num_remaining[p])


class ClosestLMExpert(BasisBasedExpert):
    def update_env(self, new_env: BaseEnv):
        """
        Update the expert's environment.

        Args:
        - new_env (BaseEnv): The new environment to set.
        """
        self.env = new_env
        self.basis = get_basis(new_env.generators)
        self.leading_terms = get_leading_terms(self.basis)

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select the pair that would yield the polynomial with leading monomial closest to any leading monomial in the basis.

        Arguments:
        - observation: A tuple (G, P) where G is the current list of generators and P is the current list of pairs.
        """

        def distance(m1: tuple[int, ...], m2: tuple[int, ...]) -> int:
            return sum(abs(a - b) for a, b in zip(m1, m2))

        G, P = observation
        leading_monomial_by_pair = lm_by_pair(self.env, G, P)
        best_pair = min(
            leading_monomial_by_pair.keys(),
            key=lambda pair: min(
                distance(leading_monomial_by_pair[pair], lt)
                for lt in self.leading_terms
            ),
            default=P[0],
        )

        return best_pair


class MCTSExpert(Expert):
    def __init__(
        self, env: BaseEnv, rollout_policy: Expert, n_simulations=50, c=1.0, gamma=0.99
    ):
        self.env = env
        self.rollout_policy = rollout_policy
        self.n_simulations = n_simulations
        self.c = c
        self.gamma = gamma

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        """
        Select the pair that would yield the polynomial with leading monomial closest to any leading monomial in the basis.

        Arguments:
        - observation: A tuple (G, P) where G is the current list of generators and P is the current list of pairs.
        """
        raise NotImplementedError()
