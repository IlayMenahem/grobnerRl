from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sympy.polys.rings import PolyElement

from grobnerRl.env import BaseEnv, GVW_buchberger, reduce, spoly


def select(
    G: Sequence[PolyElement], P: Sequence[tuple[int, int]], strategy="normal"
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
) -> tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]], BaseEnv]:
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


def _step_with_reward(
    env: BaseEnv, pair: int | tuple[int, int]
) -> tuple[list[PolyElement], list[tuple[int, int]], BaseEnv, float, bool]:
    """Like `next_step`, but also returns the env reward and a done flag."""
    new_env = copy(env)
    new_env.generators = list(env.generators)
    new_env.pairs = list(env.pairs)
    (basis, pairs), reward, terminated, truncated, _ = new_env.step(pair)
    return basis, pairs, new_env, float(reward), bool(terminated or truncated)


def _simulate_to_termination(
    env: BaseEnv, policy: "Expert", gamma: float
) -> float:
    """Roll out `policy` to termination from `env`; return γ-discounted return.

    `policy` must read its action from the observation only, not `self.env`
    (i.e. it must be a stateless-w.r.t.-env expert such as `BasicExpert`).
    Basis-based experts (`ClosestLMExpert`, `LeastRemainingPairsExpert`) are
    not valid rollout policies because they read `self.env` during `__call__`.
    """
    basis: list[PolyElement] = env.generators
    pairs: list[tuple[int, int]] = env.pairs
    total: float = 0.0
    discount: float = 1.0
    done: bool = not pairs
    while not done:
        action = policy((basis, pairs))
        basis, pairs, env, r, done = _step_with_reward(env, action)
        total += discount * r
        discount *= gamma
    return total


def get_basis(G: list[PolyElement]) -> list[PolyElement]:
    basis, _ = GVW_buchberger(G)
    return basis


def get_leading_terms(basis: list[PolyElement]) -> set[tuple[int, ...]]:
    return {p.LM for p in basis}


def lm_by_pair(
    env: BaseEnv, G: list[PolyElement], pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], tuple[int, ...]]:
    """
    Get the leading monomial of the polynomial that would be yielded by the pair reduction.

    Args:
    - env (BaseEnv): The current environment
    - G: list[PolyElement]: The current generators
    - pairs (list[tuple[int, int]]): The pairs we can reduce

    Returns:
    - dict[tuple[int, int], tuple[int, ...]]: A dictionary mapping each pair to its leading monomial.
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
            diff_monomial = tuple(abs(a - b) for a, b in zip(m1, m2))
            return G[0].ring.order(diff_monomial)

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


class RolloutExpert(Expert):
    """
    Pick the pair maximising mean γ-discounted env return when followed by
    rollouts of a base policy to termination.

    `n_rollouts > 1` is only useful for stochastic base policies (e.g.
    `BasicExpert("random")`); for deterministic bases a single rollout
    suffices.
    """

    base_policy: "Expert"
    n_rollouts: int
    gamma: float

    def __init__(
        self,
        env: BaseEnv,
        base_policy: "Expert",
        n_rollouts: int = 1,
        gamma: float = 1.0,
    ):
        super().__init__(env)
        self.base_policy = base_policy
        self.n_rollouts = n_rollouts
        self.gamma = gamma

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        _, P = observation
        if not P:
            raise ValueError("pair set must be nonempty")
        return max(P, key=self._mean_return)

    def _mean_return(self, pair: tuple[int, int]) -> float:
        total: float = 0.0
        for _ in range(self.n_rollouts):
            _, _, new_env, r, done = _step_with_reward(self.env, pair)
            future: float = (
                0.0
                if done
                else _simulate_to_termination(new_env, self.base_policy, self.gamma)
            )
            total += r + self.gamma * future
        return total / self.n_rollouts


class OptimalDPExpert(Expert):
    """
    Pick the action on the γ-optimal trajectory via memoised DFS over the
    reachable env states. Only feasible for small ideals; raises
    `RuntimeError` if the state-expansion budget is exceeded.
    """

    gamma: float
    max_states: int
    _value_cache: dict[Hashable, float]
    _action_cache: dict[Hashable, tuple[int, int]]

    def __init__(
        self,
        env: BaseEnv,
        gamma: float = 1.0,
        max_states: int = 100_000,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.max_states = max_states
        self._value_cache = {}
        self._action_cache = {}

    @staticmethod
    def _state_key(
        basis: list[PolyElement], pairs: list[tuple[int, int]]
    ) -> Hashable:
        # PolyElement has no __hash__; .terms() yields hashable (monom, coeff) tuples.
        poly_keys: tuple = tuple(tuple(p.terms()) for p in basis)
        return (poly_keys, tuple(sorted(pairs)))

    def _value(self, env: BaseEnv) -> float:
        key = self._state_key(env.generators, env.pairs)
        if key in self._value_cache:
            return self._value_cache[key]
        if len(self._value_cache) >= self.max_states:
            raise RuntimeError(
                f"OptimalDPExpert: exceeded max_states={self.max_states}"
            )
        if not env.pairs:
            self._value_cache[key] = 0.0
            return 0.0

        best_v: float = -float("inf")
        best_a: tuple[int, int] | None = None
        for pair in env.pairs:
            _, _, new_env, r, done = _step_with_reward(env, pair)
            future: float = 0.0 if done else self._value(new_env)
            v: float = r + self.gamma * future
            if v > best_v:
                best_v, best_a = v, pair

        assert best_a is not None
        self._value_cache[key] = best_v
        self._action_cache[key] = best_a
        return best_v

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        if not self.env.pairs:
            raise ValueError("pair set must be nonempty")
        self._value(self.env)
        return self._action_cache[
            self._state_key(self.env.generators, self.env.pairs)
        ]

    def update_env(self, new_env: BaseEnv):
        super().update_env(new_env)
        self._value_cache.clear()
        self._action_cache.clear()


@dataclass
class _MCTSNode:
    """A single node in the UCT tree used by `MCTSExpert`."""

    env: BaseEnv
    pairs: list[tuple[int, int]]
    terminal: bool
    visit_count: int = 0
    children: dict[tuple[int, int], "_MCTSNode"] = field(default_factory=dict)
    child_visits: dict[tuple[int, int], int] = field(default_factory=dict)
    child_value: dict[tuple[int, int], float] = field(default_factory=dict)
    child_value_sq: dict[tuple[int, int], float] = field(default_factory=dict)
    child_reward: dict[tuple[int, int], float] = field(default_factory=dict)

    def untried(self) -> list[tuple[int, int]]:
        return [p for p in self.pairs if p not in self.children]


class MCTSExpert(Expert):
    """
    Classical UCT search with rollouts to estimate action values via the env
    reward. Root action is selected by `Q(a) - var_penalty * Var(a)`, where
    `Var` is the biased (population) variance of returns at `a`.
    """

    rollout_policy: "Expert"
    n_simulations: int
    c: float
    gamma: float
    var_penalty: float

    def __init__(
        self,
        env: BaseEnv,
        rollout_policy: "Expert",
        n_simulations: int = 50,
        c: float = 1.0,
        gamma: float = 0.99,
        var_penalty: float = 0.0,
    ):
        super().__init__(env)
        self.rollout_policy = rollout_policy
        self.n_simulations = n_simulations
        self.c = c
        self.gamma = gamma
        self.var_penalty = var_penalty

    def __call__(
        self, observation: tuple[list[PolyElement], list[tuple[int, int]]]
    ) -> int | tuple[int, int]:
        _, P = observation
        if not P:
            raise ValueError("pair set must be nonempty")

        root = self._make_node(self.env)
        for _ in range(self.n_simulations):
            self._simulate(root)

        return max(root.children.keys(), key=lambda a: self._root_score(root, a))

    def _make_node(self, env: BaseEnv) -> _MCTSNode:
        pairs: list[tuple[int, int]] = list(env.pairs)
        return _MCTSNode(env=env, pairs=pairs, terminal=not pairs)

    def _ucb(self, node: _MCTSNode, action: tuple[int, int]) -> float:
        q: float = node.child_value[action] / node.child_visits[action]
        explore: float = self.c * sqrt(log(node.visit_count) / node.child_visits[action])
        return q + explore

    def _root_score(self, root: _MCTSNode, action: tuple[int, int]) -> float:
        n: int = root.child_visits[action]
        q: float = root.child_value[action] / n
        # Biased (population) variance: S/N - (W/N)^2; naturally 0 when n == 1.
        var: float = max(0.0, root.child_value_sq[action] / n - q * q)
        return q - self.var_penalty * var

    def _simulate(self, root: _MCTSNode) -> None:
        path: list[tuple[_MCTSNode, tuple[int, int]]] = []
        node: _MCTSNode = root

        # Selection + expansion.
        while True:
            if node.terminal:
                rollout_value: float = 0.0
                break

            untried: list[tuple[int, int]] = node.untried()
            if untried:
                action = untried[0]
                _, _, new_env, r, done = _step_with_reward(node.env, action)
                child = self._make_node(new_env)
                if done:
                    child.terminal = True
                node.children[action] = child
                node.child_reward[action] = r
                node.child_visits[action] = 0
                node.child_value[action] = 0.0
                node.child_value_sq[action] = 0.0
                path.append((node, action))
                rollout_value = (
                    0.0
                    if child.terminal
                    else _simulate_to_termination(
                        child.env, self.rollout_policy, self.gamma
                    )
                )
                break

            action = max(node.pairs, key=lambda a: self._ucb(node, a))
            path.append((node, action))
            node = node.children[action]

        # Backup.
        cumulative: float = rollout_value
        for parent, action in reversed(path):
            cumulative = parent.child_reward[action] + self.gamma * cumulative
            parent.child_visits[action] += 1
            parent.child_value[action] += cumulative
            parent.child_value_sq[action] += cumulative * cumulative
            parent.visit_count += 1
