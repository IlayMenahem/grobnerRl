"""
implementation of deep buchberger environment with gaubmoller criteria, f4, and f5
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from sympy.polys.rings import PolyElement

from grobnerRl.envs.ideals import IdealGenerator


def tokenize(ideal: Sequence[PolyElement]) -> list[ArrayLike]:
    """
    takes an ideal and returns a tokenized version of it, a list of arrays, each of the arrays
    representing a polynomial monomials

    Parameters:
    ideal: list[PolyElement] - The ideal generators to be tokenized

    Returns: tokenized ideal
    """
    polys_monomials = [
        np.concat(
            (
                np.array(list(map(int, poly.coeffs()))).reshape((1, -1)).T,
                np.array(poly.monoms()),
            ),
            axis=1,
        )
        for poly in ideal
        if poly != 0
    ]

    return polys_monomials


def make_obs(G, P) -> tuple[list[ArrayLike], list[tuple[int, int]]]:
    P = deepcopy(P)
    G = tokenize(G)

    return G, P


def spoly(f: PolyElement, g: PolyElement) -> PolyElement:
    """
    Compute the S-polynomial of f and g.

    Args:
    - f: A polynomial.
    - g: A polynomial.

    Returns:
    - The S-polynomial of f and g.
    """
    lmf = f.LM
    lmg = g.LM

    R = f.ring
    lcm = R.monomial_lcm(lmf, lmg)

    s1 = f.mul_monom(R.monomial_div(lcm, lmf))
    s2 = g.mul_monom(R.monomial_div(lcm, lmg))

    return s1 - s2


def reduce(g: PolyElement, F: list[PolyElement]) -> tuple[PolyElement, dict]:
    """
    Return a remainder and stats when g is divided by monic polynomials F.

    Args:
    - g: Dividend polynomial.
    - F: List of monic divisor polynomials.

    Returns:
    A tuple containing:
    - Remainder polynomial
    - Dictionary with statistics (e.g., 'steps': number of reduction steps)

    Example:
        >>> import sympy as sp
        >>> R, a, b, c, d = sp.ring('a,b,c,d', sp.QQ, 'lex')
        >>> reduce(a**3*b*c**2 + a**2*c, [a**2 + b, a*b*c + c, a*c**2 + b**2])
        (b*c**2 - b*c, {'steps': 3})
    """

    ring = g.ring
    monomial_div = ring.monomial_div
    lmF = [f.LM for f in F]

    stats = {"steps": 0}
    r = ring.zero
    h = g.copy()

    while h:
        lmh, lch = h.LT
        found_divisor = False

        for f, lmf in zip(F, lmF):
            m = monomial_div(lmh, lmf)
            if m is not None:
                h = h - f.mul_term((m, lch))
                found_divisor = True
                stats["steps"] += 1
                break

        if not found_divisor:
            if lmh in r:
                r[lmh] += lch
            else:
                r[lmh] = lch
            del h[lmh]

    return r, stats


def update(
    G: list[PolyElement], P: list[tuple[int, int]], f: PolyElement
) -> tuple[list[PolyElement], list[tuple[int, int]]]:
    """
    Return the updated lists of polynomials and pairs when f is added to the basis G.

    The inputs G and P are modified by this function.

    Args:
    - G: Current list of polynomial generators.
    - P: Current list of s-pairs.
    - f: New polynomial to add to the basis.

    Returns:
    A tuple containing:
    - Updated list of polynomial generators
    - Updated list of s-pairs
    """

    lmf = f.LM
    R = f.ring
    lcm = R.monomial_lcm
    mul = R.monomial_mul
    div = R.monomial_div

    lmG = [g.LM for g in G]
    new_index = len(G)

    def is_chain_redundant(pair: tuple[int, int]) -> bool:
        i, j = pair
        gamma = lcm(lmG[i], lmG[j])
        if div(gamma, lmf) is None:
            return False
        return gamma != lcm(lmG[i], lmf) and gamma != lcm(lmG[j], lmf)

    def is_lcm_redundant(pair: tuple[int, int]) -> bool:
        i, j = pair
        return lcm(lmG[i], lmG[j]) == mul(lmG[i], lmG[j])

    P[:] = [
        pair for pair in P if not (is_lcm_redundant(pair) or is_chain_redundant(pair))
    ]

    indices_by_lcm = defaultdict(list)
    for idx, lm in enumerate(lmG):
        indices_by_lcm[lcm(lm, lmf)].append(idx)

    minimal_lcms = []
    new_pairs = []
    for gamma in sorted(indices_by_lcm, key=R.order):
        if any(div(gamma, existing) is not None for existing in minimal_lcms):
            continue
        minimal_lcms.append(gamma)

        if any(lcm(lmG[i], lmf) == mul(lmG[i], lmf) for i in indices_by_lcm[gamma]):
            continue
        new_pairs.append((indices_by_lcm[gamma][0], new_index))

    new_pairs.sort(key=lambda pair: pair[0])

    G.append(f)
    P.extend(new_pairs)

    return G, P


def minimalize(G: list[PolyElement]) -> list[PolyElement]:
    """
    Return the minimal Groebner basis from Groebner basis G.

    Args:
    - G: A Groebner basis.

    Returns:
    - The minimal Groebner basis.
    """

    if len(G) > 0:
        R = G[0].ring
    else:
        return []

    Gmin = []
    for f in sorted(G, key=lambda h: R.order(h.LM)):
        if all(not R.monomial_div(f.LM, g.LM) for g in Gmin):
            Gmin.append(f)

    return Gmin


def interreduce(G):
    """
    Return the interreduced Groebner basis from Groebner basis G.

    Args:
    - G: A Groebner basis.

    Returns:
    - The interreduced Groebner basis.
    """

    return [G[i].rem(G[:i] + G[i + 1 :]).monic() for i in range(len(G))]


# GVW Signature-based Algorithm Helper Functions


def _signature_mul(
    signature: tuple[tuple[int, ...], int], mon: tuple[int, ...], mul_fn: Callable
) -> tuple[tuple[int, ...], int]:
    """Multiply a signature by a monomial."""
    return (mul_fn(signature[0], mon), signature[1])


def _signature_key(
    signature: tuple[tuple[int, ...], int],
    initial_lms: list[tuple[int, ...]],
    mul_fn: Callable,
    order_fn: Callable,
) -> tuple:
    """Compute the ordering key for a signature."""
    mon, idx = signature
    lead = mul_fn(mon, initial_lms[idx])
    return (order_fn(lead), idx, order_fn(mon))


def _signature_lt(
    left: tuple[tuple[int, ...], int],
    right: tuple[tuple[int, ...], int],
    initial_lms: list[tuple[int, ...]],
    mul_fn: Callable,
    order_fn: Callable,
) -> bool:
    """Check if left signature is less than right signature."""
    return _signature_key(left, initial_lms, mul_fn, order_fn) < _signature_key(
        right, initial_lms, mul_fn, order_fn
    )


def _signature_divides(
    base: tuple[tuple[int, ...], int],
    target: tuple[tuple[int, ...], int],
    div_fn: Callable,
) -> bool:
    """Check if base signature divides target signature."""
    if base[1] != target[1]:
        return False
    return div_fn(target[0], base[0]) is not None


def _is_blocked(
    signature: tuple[tuple[int, ...], int],
    syzygies: set[tuple[tuple[int, ...], int]],
    div_fn: Callable,
) -> bool:
    """Check if a signature is blocked by any syzygy."""
    return any(_signature_divides(blocker, signature, div_fn) for blocker in syzygies)


def _regular_top_reduce(
    signature: tuple[tuple[int, ...], int],
    poly: PolyElement,
    signatures: list[tuple[tuple[int, ...], int]],
    generators: list[PolyElement],
    initial_lms: list[tuple[int, ...]],
    div_fn: Callable,
    mul_fn: Callable,
    order_fn: Callable,
) -> PolyElement:
    """Perform regular top reduction on a polynomial."""
    if not signatures:
        return poly

    while poly != 0:
        lm_poly, lc_poly = poly.LT
        reduced = False
        for sig_reducer, reducer in zip(signatures, generators):
            mult = div_fn(lm_poly, reducer.LM)
            if mult is None:
                continue
            sig_mult = _signature_mul(sig_reducer, mult, mul_fn)
            if _signature_lt(sig_mult, signature, initial_lms, mul_fn, order_fn):
                ratio = lc_poly / reducer.LC
                poly = poly - reducer.mul_term((mult, ratio))
                reduced = True
                break
        if not reduced:
            break

    return poly


def _is_super_top_reducible(
    signature: tuple[tuple[int, ...], int],
    poly: PolyElement,
    signatures: list[tuple[tuple[int, ...], int]],
    generators: list[PolyElement],
    div_fn: Callable,
    mul_fn: Callable,
) -> bool:
    """Check if a polynomial is super top reducible."""
    if not signatures or poly == 0:
        return False

    lm_poly = poly.LM
    for sig_reducer, reducer in zip(signatures, generators):
        mult = div_fn(lm_poly, reducer.LM)
        if mult is None:
            continue
        if _signature_mul(sig_reducer, mult, mul_fn) == signature:
            return True
    return False


def _remove_multiples(
    jpairs: list[tuple[tuple[tuple[int, ...], int], PolyElement]],
    pairs: list[tuple[tuple[int, ...], int]],
    blockers: Sequence[tuple[tuple[int, ...], int]],
    div_fn: Callable,
) -> tuple[
    list[tuple[tuple[tuple[int, ...], int], PolyElement]],
    list[tuple[tuple[int, ...], int]],
]:
    """Remove pairs that are multiples of blockers."""
    if not blockers:
        return jpairs, pairs

    filtered_jpairs: list[tuple[tuple[tuple[int, ...], int], PolyElement]] = []
    filtered_pairs: list[tuple[tuple[int, ...], int]] = []

    for sig, poly in jpairs:
        if any(_signature_divides(blocker, sig, div_fn) for blocker in blockers):
            continue
        filtered_jpairs.append((sig, poly))
        filtered_pairs.append(sig)

    return filtered_jpairs, filtered_pairs


def _add_j_pair(
    jpairs: list[tuple[tuple[tuple[int, ...], int], PolyElement]],
    pairs: list[tuple[tuple[int, ...], int]],
    signature: tuple[tuple[int, ...], int],
    poly: PolyElement,
    syzygies: set[tuple[tuple[int, ...], int]],
    div_fn: Callable,
    order_fn: Callable,
) -> tuple[
    list[tuple[tuple[tuple[int, ...], int], PolyElement]],
    list[tuple[tuple[int, ...], int]],
]:
    """Add a new j-pair, replacing existing one if signature matches and new poly has smaller LM."""
    if _is_blocked(signature, syzygies, div_fn):
        return jpairs, pairs

    poly_lm_key = order_fn(poly.LM)
    for idx, (existing_sig, existing_poly) in enumerate(jpairs):
        if existing_sig == signature:
            if order_fn(existing_poly.LM) > poly_lm_key:
                jpairs[idx] = (signature, poly)
                pairs[idx] = signature
            return jpairs, pairs

    jpairs.append((signature, poly))
    pairs.append(signature)
    return jpairs, pairs


def _initialize_gvw_state(inputs: Sequence[PolyElement]) -> dict[str, Any]:
    """Initialize the GVW algorithm state from input polynomials."""
    state = {
        "generators": [],
        "pairs": [],
        "syzygies": set(),
        "jpairs": [],
        "signatures": [],
        "initial_lms": [],
        "ring": None,
        "mul": None,
        "div": None,
        "lcm": None,
        "order": None,
        "zero_monom": (),
    }

    monic_inputs = [poly.monic() for poly in inputs if poly != 0]
    if not monic_inputs:
        return state

    ring = monic_inputs[0].ring
    state["ring"] = ring
    state["mul"] = ring.monomial_mul
    state["div"] = ring.monomial_div
    state["lcm"] = ring.monomial_lcm
    state["order"] = ring.order
    state["zero_monom"] = (0,) * ring.ngens
    state["initial_lms"] = [poly.LM for poly in monic_inputs]

    state["jpairs"] = [
        ((state["zero_monom"], idx), poly) for idx, poly in enumerate(monic_inputs)
    ]
    state["pairs"] = [signature for signature, _ in state["jpairs"]]

    return state


def _select_min_signature_index(state: dict[str, Any]) -> int:
    """Select the index of the pair with minimal signature."""
    if not state["jpairs"]:
        raise ValueError("no pairs to select")
    return min(
        (_signature_key(sig, state["initial_lms"], state["mul"], state["order"]), idx)
        for idx, (sig, _) in enumerate(state["jpairs"])
    )[1]


def _process_pair(state: dict[str, Any], index: int) -> dict[str, Any]:
    """Process a signature pair at the given index."""
    if index < 0 or index >= len(state["jpairs"]):
        raise IndexError("pair index out of range")

    lcm_fn = state["lcm"]
    div_fn = state["div"]
    mul_fn = state["mul"]
    order_fn = state["order"]

    signature, poly = state["jpairs"].pop(index)
    state["pairs"].pop(index)

    result: dict[str, Any] = {
        "signature": signature,
        "poly_added": False,
        "syzygy_added": False,
    }

    if _is_blocked(signature, state["syzygies"], div_fn):
        return result

    working_poly = poly.copy()
    working_poly = _regular_top_reduce(
        signature,
        working_poly,
        state["signatures"],
        state["generators"],
        state["initial_lms"],
        div_fn,
        mul_fn,
        order_fn,
    )

    if working_poly == 0:
        state["syzygies"].add(signature)
        state["jpairs"], state["pairs"] = _remove_multiples(
            state["jpairs"], state["pairs"], [signature], div_fn
        )
        result["syzygy_added"] = True
        return result

    if _is_super_top_reducible(
        signature,
        working_poly,
        state["signatures"],
        state["generators"],
        div_fn,
        mul_fn,
    ):
        return result

    working_poly = working_poly.monic()

    new_syzygies: list[tuple[tuple[int, ...], int]] = []
    for sig_existing, existing_poly in zip(state["signatures"], state["generators"]):
        sig_from_existing = _signature_mul(sig_existing, working_poly.LM, mul_fn)
        sig_from_new = _signature_mul(signature, existing_poly.LM, mul_fn)
        leading_sig = (
            sig_from_existing
            if _signature_lt(
                sig_from_new, sig_from_existing, state["initial_lms"], mul_fn, order_fn
            )
            else sig_from_new
        )
        if leading_sig not in state["syzygies"]:
            new_syzygies.append(leading_sig)

    if new_syzygies:
        state["syzygies"].update(new_syzygies)
        state["jpairs"], state["pairs"] = _remove_multiples(
            state["jpairs"], state["pairs"], new_syzygies, div_fn
        )

    current_signatures = list(state["signatures"])
    current_generators = list(state["generators"])
    for sig_existing, existing_poly in zip(current_signatures, current_generators):
        lcm_mon = lcm_fn(working_poly.LM, existing_poly.LM)
        mult_new = div_fn(lcm_mon, working_poly.LM)
        mult_existing = div_fn(lcm_mon, existing_poly.LM)
        if mult_new is None or mult_existing is None:
            continue

        sig_new_side = _signature_mul(signature, mult_new, mul_fn)
        sig_existing_side = _signature_mul(sig_existing, mult_existing, mul_fn)

        if _signature_lt(
            sig_new_side, sig_existing_side, state["initial_lms"], mul_fn, order_fn
        ):
            candidate_sig = sig_existing_side
            candidate_poly = existing_poly.mul_term((mult_existing, 1))
        else:
            candidate_sig = sig_new_side
            candidate_poly = working_poly.mul_term((mult_new, 1))

        state["jpairs"], state["pairs"] = _add_j_pair(
            state["jpairs"],
            state["pairs"],
            candidate_sig,
            candidate_poly,
            state["syzygies"],
            div_fn,
            order_fn,
        )

    state["signatures"].append(signature)
    state["generators"].append(working_poly)
    result["poly_added"] = True

    return result


def select(
    G: list[PolyElement], P: list[tuple[int, int]], strategy="normal"
) -> tuple[int, int]:
    """
    Select and return a pair from P.

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
        elif s == "degree_after_reduce":
            after_red, _ = reduce(spoly(G[p[0]], G[p[1]]), G)

            if after_red == 0:
                return (np.inf, ())

            return R.order(after_red.LM)
        else:
            raise ValueError("unknown selection strategy")

    return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))


def buchberger(G: list[PolyElement]) -> tuple[list[PolyElement], dict]:
    """
    Compute the Groebner basis of the ideal generated by G using Buchberger's algorithm.

    Args:
    - G: List of polynomial generators.

    Returns:
    - Tuple where the first element is a minimal, interreduced Groebner basis of the
      ideal generated by G, and the second element is a dictionary with run statistics.
    """

    if not G:
        return [], {
            "zero_reductions": 0,
            "nonzero_reductions": 0,
            "total_reduction_steps": 0,
            "pairs_processed": 0,
        }

    basis: list[PolyElement] = []
    pairs: list[tuple[int, int]] = []

    for poly in G:
        update(basis, pairs, poly.monic())

    stats = {
        "zero_reductions": 0,
        "nonzero_reductions": 0,
        "total_reduction_steps": 0,
        "pairs_processed": 0,
    }

    while pairs:
        i, j = select(basis, pairs)
        pairs.remove((i, j))
        stats["pairs_processed"] += 1

        s_poly = spoly(basis[i], basis[j])
        remainder, reduction_stats = reduce(s_poly, basis)
        stats["total_reduction_steps"] += reduction_stats.get("steps", 0)

        if remainder != 0:
            stats["nonzero_reductions"] += 1
            update(basis, pairs, remainder.monic())
        else:
            stats["zero_reductions"] += 1

    reduced_basis = minimalize(basis)
    reduced_basis = interreduce(reduced_basis) if reduced_basis else []

    return reduced_basis, stats


def GVW_buchberger(
    G: list[PolyElement],
) -> tuple[list[PolyElement], list[tuple[tuple[int, ...], int]]]:
    """
    Compute a Groebner basis using the GVW signature-based Buchberger algorithm with the g2 module order.

    Args:
    - G: List of polynomial generators.

    Returns:
    - Tuple where the first element is a minimal, interreduced Groebner basis and the second element is a
      list of leading module terms of discovered syzygies (each encoded as (monomial, index)).
    """

    state = _initialize_gvw_state(G)

    while state["jpairs"]:
        index = _select_min_signature_index(state)
        _process_pair(state, index)

    basis = interreduce(minimalize(state["generators"])) if state["generators"] else []

    syzygies = []
    if state["syzygies"]:
        syzygies = sorted(
            state["syzygies"],
            key=lambda sig: _signature_key(
                sig, state["initial_lms"], state["mul"], state["order"]
            ),
        )

    return basis, syzygies


class BaseEnv(ABC):
    generators: list[PolyElement]
    pairs: list[tuple[int, int]]

    @abstractmethod
    def __init__(self, ideal_generator: IdealGenerator):
        """
        Initialize the base environment.

        Parameters:
        ideal_generator: IdealGenerator - Generator for ideals to be used in the environment.
        """
        pass

    @abstractmethod
    def reset(
        self, seed=None, options=None
    ) -> tuple[tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]]], dict]:
        """
        Reset the environment to start a new episode.

        Parameters:
        seed: Optional seed for random number generation.
        options: Optional additional options.

        Returns:
        observation: The initial observation for the environment.
        info: Additional information about the environment state.
        """
        pass

    @abstractmethod
    def step(
        self, action: int | tuple[int, int]
    ) -> tuple[
        tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]]],
        int,
        bool,
        bool,
        dict,
    ]:
        """
        Take a step in the environment based on the given action.

        Parameters:
        action: int | tuple[int, int] - The action to take (either an integer or a pair of indices).

        Returns:
        observation: The new observation after taking the action.
        reward: The reward received after taking the action.
        terminated: Boolean indicating if the episode has terminated.
        truncated: Boolean indicating if the episode has been truncated.
        info: Additional information about the environment state.
        """
        pass


class BuchbergerEnv(BaseEnv):
    generators: list[PolyElement]
    pairs: list[tuple[int, int]]

    def __init__(self, ideal_generator: IdealGenerator, mode="eval"):
        """
        Initialize the Buchberger environment.

        Parameters:
        ideal_generator: IdealGenerator - Generator for ideals to be used in the environment.
        mode: str - Mode of the environment ('train' or 'eval').
        """
        super().__init__(ideal_generator)
        self.ideal_generator = ideal_generator
        self.mode = mode

        self.generators = []
        self.pairs = []

    def reset(
        self, seed=None, options=None
    ) -> tuple[tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]]], dict]:
        """
        Reset the environment to start a new episode.

        Parameters:
        seed: Optional seed for random number generation.
        options: Optional additional options.

        Returns:
        observation: The initial observation for the environment.
        info: Additional information about the environment state.
        """
        self.generators = []
        self.pairs = []
        generators = next(self.ideal_generator)

        for g in generators:
            self.generators, self.pairs = update(self.generators, self.pairs, g.monic())

        observation = (self.generators, self.pairs)
        if self.mode == "train":
            observation = make_obs(*observation)

        return observation, {}

    def step(
        self, action: int | tuple[int, int]
    ) -> tuple[
        tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]]],
        int,
        bool,
        bool,
        dict,
    ]:
        """
        Take a step in the environment based on the given action.

        Parameters:
        action: int | tuple[int, int] - The action to take (either an integer or a pair of indices).

        Returns:
        observation: The new observation after taking the action.
        reward: The reward received after taking the action.
        terminated: Boolean indicating if the episode has terminated.
        truncated: Boolean indicating if the episode has been truncated.
        info: Additional information about the environment state.
        """

        def int_to_pair(action: int) -> tuple[int, int]:
            return (action // len(self.generators), action % len(self.generators))

        if isinstance(action, int):
            action = int_to_pair(action)

        self.pairs.remove(action)

        # Compute the S-polynomial, and if non-zero after reduction, update the basis
        poly = spoly(self.generators[action[0]], self.generators[action[1]])
        poly, _ = reduce(poly, self.generators)
        if poly != 0:
            self.generators, self.pairs = update(
                self.generators, self.pairs, poly.monic()
            )

        terminated = len(self.pairs) == 0
        reward = -1 if not terminated else 0
        truncated = False

        observation = (self.generators, self.pairs)
        if self.mode == "train":
            observation = make_obs(*observation)

        return observation, reward, terminated, truncated, {}


class GVWEnv(BaseEnv):
    """Gymnasium environment wrapping the GVW signature-based Buchberger algorithm."""

    def __init__(self, ideal_generator: IdealGenerator, mode="eval"):
        super().__init__(ideal_generator)
        self.ideal_generator = ideal_generator
        self.mode = mode

        self._state: dict[str, Any] = {}
        self.generators: list[PolyElement] = []
        self.pairs: list[tuple[tuple[int, ...], int]] = []

    def _current_observation(
        self,
    ) -> tuple[list[PolyElement] | list[ArrayLike], list[tuple[int, int]]]:
        observation = (self.generators, self.pairs)
        if self.mode == "train":
            observation = make_obs(*observation)
        return observation

    def _reset_internal_state(self) -> None:
        self._state = {}
        self.generators = []
        self.pairs = []

    def reset(self, *, seed=None, options=None):
        self._reset_internal_state()

        if seed is not None and hasattr(self.ideal_generator, "seed"):
            self.ideal_generator.seed(seed)

        generators = next(self.ideal_generator)
        self._state = _initialize_gvw_state(generators)
        self.generators = self._state["generators"]
        self.pairs = self._state["pairs"]

        return self._current_observation(), {}

    def _resolve_action(
        self, action: int | tuple[int, int] | tuple[tuple[int, ...], int]
    ) -> int:
        if isinstance(action, int):
            index = action
        elif (
            isinstance(action, tuple)
            and len(action) == 2
            and isinstance(action[0], Sequence)
            and not isinstance(action[0], (int, np.integer))
        ):
            signature = (tuple(action[0]), action[1])
            if signature not in self.pairs:
                raise ValueError("signature not found in current pair list")
            index = self.pairs.index(signature)
        else:
            raise TypeError("action must be an index or signature tuple")

        if index < 0 or index >= len(self.pairs):
            raise IndexError("action index out of range")

        return index

    def step(
        self, action: int | tuple[int, int]
    ) -> tuple[
        tuple[list[ArrayLike] | list[PolyElement], list[tuple[int, int]]],
        int,
        bool,
        bool,
        dict,
    ]:
        """
        Take a step in the environment based on the given action.

        Parameters:
        action: int | tuple[int, int] - The action to take (either an integer or a signature tuple).

        Returns:
        - A tuple containing the new observation, reward, termination status, truncation status, and additional info.
        """
        if not self._state.get("jpairs"):
            raise ValueError("no pairs available to process")

        index = self._resolve_action(action)
        _process_pair(self._state, index)

        # Update instance variables to reflect state changes
        self.generators = self._state["generators"]
        self.pairs = self._state["pairs"]

        reward = -1
        terminated = not bool(self._state.get("jpairs"))

        return self._current_observation(), reward, terminated, False, {}
