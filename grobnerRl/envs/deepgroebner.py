"""
An environment for computing Groebner bases with Buchberger's algorithm.

credit to the authors of the deepgroebner paper
"""

import numpy as np
from copy import deepcopy
from sympy.polys.rings import PolyElement
from collections.abc import Sequence
import bisect
import gymnasium as gym
from grobnerRl.envs.ideals import IdealGenerator, parse_ideal_dist


def tokenize(ideal: Sequence[PolyElement]) -> list[np.ndarray]:
    '''
    takes an ideal and returns a tokenized version of it, a list of arrays, each of the arrays
    representing a polynomial monomials

    Parameters:
    ideal: list[PolyElement] - The ideal generators to be tokenized

    Returns: tokenized ideal
    '''
    polys_monomials = [poly.monoms() for poly in ideal]

    return polys_monomials


def make_obs(G, P) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    P = deepcopy(P)
    G = tokenize(G)

    return G, P


def spoly(f, g, lmf=None, lmg=None):
    """Return the s-polynomial of monic polynomials f and g."""
    lmf = f.LM if lmf is None else lmf
    lmg = g.LM if lmg is None else lmg
    R = f.ring
    lcm = R.monomial_lcm(lmf, lmg)
    s1 = f.mul_monom(R.monomial_div(lcm, lmf))
    s2 = g.mul_monom(R.monomial_div(lcm, lmg))
    return s1 - s2


def reduce(g, F, lmF=None):
    """Return a remainder and stats when g is divided by monic polynomials F.

    Parameters
    ----------
    g : polynomial
        Dividend polynomial.
    F : list
        List of monic divisor polynomials.
    lmF : list, optional
        Precomputed list of lead monomials of F for efficiency.

    Examples
    --------
    >>> import sympy as sp
    >>> R, a, b, c, d = sp.ring('a,b,c,d', sp.QQ, 'lex')
    >>> reduce(a**3*b*c**2 + a**2*c, [a**2 + b, a*b*c + c, a*c**2 + b**2])
    (b*c**2 - b*c, {'steps': 3})

    """
    ring = g.ring
    monomial_div = ring.monomial_div
    lmF = [f.LM for f in F] if lmF is None else lmF

    stats = {'steps': 0}
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
                stats['steps'] += 1
                break

        if not found_divisor:
            if lmh in r:
                r[lmh] += lch
            else:
                r[lmh] = lch
            del h[lmh]

    return r, stats


def update(G, P, f, strategy='gebauermoeller', lmG=None):
    """Return the updated lists of polynomials and pairs when f is added to the basis G.

    The inputs G and P are modified by this function.

    Parameters
    ----------
    G : list
        Current list of polynomial generators.
    P : list
        Current list of s-pairs.
    f : polynomial
        New polynomial to add to the basis.
    strategy : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.

        Strategy can be 'none' (eliminate no pairs), 'lcm' (only eliminate pairs that
        fail the LCM criterion), or 'gebauermoeller' (use full Gebauer-Moeller elimination).
    lmG : list, optional
        Precomputed list of the lead monomials of G for efficiency.

    Examples
    --------
    >>> import sympy as sp
    >>> R, x, y, z = sp.ring('x,y,z', sp.FF(32003), 'grevlex')
    >>> G = [x*y**2 + 2*z, x*z**2 - y**2 - z, x + 3]
    >>> P = [(0, 2)]
    >>> f = y**2*z**3 + 4*z**4 - y**2 + z**2
    >>> update(G, P, f)
    ([x*y**2 + 2 mod 32003*z,
      x*z**2 + 32002 mod 32003*y**2 + 32002 mod 32003*z,
      x + 3 mod 32003,
      y**2*z**3 + 4 mod 32003*z**4 + 32002 mod 32003*y**2 + z**2],
     [(0, 2)])

    """
    def chain_rule(p, lmG):
        i, j = p
        gam = lcm(lmG[i], lmG[j])

        chain_rule_applicable = (div(gam, lmf) and gam != lcm(lmG[i], lmf) and gam != lcm(lmG[j], lmf))

        return chain_rule_applicable

    def power_rule(p, G):
        i, j = p

        i_lm = G[i].LM
        j_lm = G[j].LM

        power_rule_applicable = (lcm(i_lm, j_lm) == mul(i_lm, j_lm))

        return power_rule_applicable

    lmf = f.LM
    lmG = [g.LM for g in G] if lmG is None else lmG
    R = f.ring
    lcm = R.monomial_lcm
    mul = R.monomial_mul
    div = R.monomial_div
    m = len(G)

    if strategy == 'none':
        P_ = [(i, m) for i in range(m)]

    elif strategy == 'lcm':
        P_ = [(i, m) for i in range(m) if lcm(lmG[i], lmf) != mul(lmG[i], lmf)]

    elif strategy == 'gebauermoeller':
        P[:] = [p for p in P if not chain_rule(p, lmG)]

        lcms = {}
        for i in range(m):
            lcms.setdefault(lcm(lmG[i], lmf), []).append(i)
        min_lcms = []
        P_ = []
        for gam in sorted(lcms.keys(), key=R.order):
            if all(not div(gam, m) for m in min_lcms):
                min_lcms.append(gam)
                if not any(lcm(lmG[i], lmf) == mul(lmG[i], lmf) for i in lcms[gam]):
                    P_.append((lcms[gam][0], m))
        P_.sort(key=lambda p: p[0])

    elif strategy == 'optimal':
        def none_zero(p):
            i, j = p
            s = spoly(G[i], G[j])
            r, _ = reduce(s, G)

            return r != 0

        G, P = update(G, P, f, strategy='gebauermoeller')
        P = [pair for pair in P if none_zero(pair) and not power_rule(pair, G)]

        return G, P

    else:
        raise ValueError('unknown elimination strategy')

    G.append(f)
    P.extend(P_)

    return G, P


def minimalize(G):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    R = G[0].ring if len(G) > 0 else None
    Gmin = []
    for f in sorted(G, key=lambda h: R.order(h.LM)):
        if all(not R.monomial_div(f.LM, g.LM) for g in Gmin):
            Gmin.append(f)
    return Gmin


def interreduce(G):
    """Return the reduced Groebner basis from minimal Groebner basis G."""
    Gred = []
    for i in range(len(G)):
        g = G[i].rem(G[:i] + G[i+1:])
        Gred.append(g.monic())
    return Gred


def select(G, P, strategy='normal'):
    """Select and return a pair from P."""
    if not len(G) > 0:
        raise ValueError('polynomial list must be nonempty')

    if not len(P) > 0:
        raise ValueError('pair set must be nonempty')

    R = G[0].ring

    if isinstance(strategy, str):
        strategy = [strategy]

    def strategy_key(p, s):
        """Return a sort key for pair p in the strategy s."""

        if s == 'first':
            return p[1], p[0]
        elif s == 'normal':
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return R.order(lcm)
        elif s == 'degree':
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return sum(lcm)
        elif s == 'random':
            return np.random.rand()
        elif s == 'degree_after_reduce':
            after_red, _ = reduce(spoly(G[p[0]], G[p[1]]), G)

            if after_red == 0:
                return (np.inf, ())

            return R.order(after_red.LM)
        else:
            raise ValueError('unknown selection strategy')

    return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))


def buchberger(F, S=None, elimination='gebauermoeller', selection='normal', rewards='additions', sort_reducers=True, gamma=0.99):
    """Return the Groebner basis for the ideal generated by F using Buchberger's algorithm.

    Parameters
    ----------
    F : list
        List of polynomial generators.
    S : list or None, optional
        List of current remaining s-pairs (None indicates no s-pair has been done yet).
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.
    rewards : {'additions', 'reductions'}, optional
        Reward value for each step.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order by lead monomial.
    gamma : float, optional
        Discount rate for rewards.

    """

    if S is None:
        G, lmG = [], []
        P = []
        for f in F:
            G, P = update(G, P, f.monic(), strategy=elimination)
            lmG.append(f.LM)
    else:
        G, lmG = F, [f.LM for f in F]
        P = S

    stats = {'zero_reductions': 0,
             'nonzero_reductions': 0,
             'polynomial_additions': 0,
             'total_reward': 0.0,
             'discounted_return': 0.0}
    discount = 1.0

    if sort_reducers and len(G) > 0:
        order = G[0].ring.order
        G_ = [g.copy() for g in G]
        G_.sort(key=lambda g: order(g.LM))
        lmG_, keysG_ = [g.LM for g in G_], [order(g.LM) for g in G_]
    else:
        G_, lmG_ = G, lmG

    while P:
        i, j = select(G, P, strategy=selection)
        P.remove((i, j))
        s = spoly(G[i], G[j], lmf=lmG[i], lmg=lmG[j])
        r, s = reduce(s, G_)
        reward = (-1.0 - s['steps']) if rewards == 'additions' else -1.0
        stats['polynomial_additions'] += s['steps'] + 1
        stats['total_reward'] += reward
        stats['discounted_return'] += discount * reward
        discount *= gamma
        if r != 0:
            G, P = update(G, P, r.monic(), lmG=lmG, strategy=elimination)
            lmG.append(r.LM)
            if sort_reducers:
                key = order(r.LM)
                index = bisect.bisect(keysG_, key)
                G_.insert(index, r.monic())
                lmG_.insert(index, r.LM)
                keysG_.insert(index, key)
            else:
                G_ = G
                lmG_ = lmG
            stats['nonzero_reductions'] += 1
        else:
            stats['zero_reductions'] += 1

    return interreduce(minimalize(G)), stats


class BuchbergerAgent:
    """
    An agent that follows standard selection strategies.

    Parameters
    ----------
    selection : {'normal', 'first', 'degree', 'random', 'degree_after_reduce'}
        The selection strategy used to pick pairs.

    """

    def __init__(self, selection='normal'):
        self.strategy = selection

    def act(self, state):
        G, P = state
        return select(G, P, strategy=self.strategy)


class BuchbergerEnv(gym.Env):
    """
    A Gymnasium environment for computing Groebner bases using Buchberger's algorithm.

    This environment provides a reinforcement learning interface for the Groebner basis
    computation problem. At each step, the agent selects a pair of polynomials to
    reduce, and the environment returns the updated state and a reward.

    Parameters
    ----------
    ideal_dist : str or IdealGenerator, optional
        IdealGenerator or string naming the ideal distribution.
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.
    rewards : {'additions', 'reductions'}, optional
        Reward value for each step.
    sort_input : bool, optional
        Whether to sort the initial generating set by lead monomial.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order by lead monomial.
    mode : {'train', 'eval'}, optional
        Mode for the environment. In 'train' mode, actions are integers and ideals are tokenized.
    """

    def __init__(self, ideal_dist='3-20-10-uniform', elimination='gebauermoeller',
                 rewards='additions', sort_input=False, sort_reducers=True, mode='eval'):

        self.mode = mode
        self.ideal_gen = self._make_ideal_gen(ideal_dist)
        self.elimination = elimination
        self.rewards = rewards
        self.sort_input = sort_input
        self.sort_reducers = sort_reducers

    def reset(self, seed=None, options=None):
        """Initialize the polynomial list and pair list for a new Groebner basis computation."""
        if seed is not None:
            self.seed(seed)

        F = next(self.ideal_gen)
        self.order = F[0].ring.order
        if self.sort_input:
            F.sort(key=lambda f: self.order(f.LM))

        self.G, self.lmG = [], []                     # the generators in inserted order
        self.G_, self.lmG_, self.keysG_ = [], [], []  # the reducers if sort_reducers
        self.P = []                                   # the pair set

        for f in F:
            self.G, self.P = update(self.G, self.P, f.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(f.LM)
            if self.sort_reducers:
                key = self.order(f.LM)
                index = bisect.bisect(self.keysG_, key)
                self.G_.insert(index, f.monic())
                self.lmG_.insert(index, f.LM)
                self.keysG_.insert(index, key)
            else:
                self.G_ = self.G
                self.lmG_ = self.lmG

        observation = (self.G, self.P)
        if self.mode == 'train':
            observation = make_obs(*observation)

        info = {}

        if not self.P:
            return self.reset()

        return observation, info

    def step(self, action: int|tuple[int,int]):
        """Perform one reduction and return the new polynomial list and pair list."""
        def int_action_to_pair(action: int) -> tuple[int, int]:
            return (action // len(self.G), action % len(self.G))

        if self.mode == 'train' and isinstance(action, (int, np.integer)):
            action = int_action_to_pair(action)

        if action not in self.P:
            observation = (self.G, self.P)
            if self.mode == 'train':
                observation = make_obs(*observation)

            reward = -1.0
            terminated = False
            truncated = False
            info = {'invalid_action': True}

            return observation, reward, terminated, truncated, info

        i, j = action

        self.P.remove(action)
        s = spoly(self.G[i], self.G[j], lmf=self.lmG[i], lmg=self.lmG[j])
        r, stats = reduce(s, self.G_, lmF=self.lmG_)

        if r != 0:
            self.G, self.P = update(self.G, self.P, r.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(r.LM)
            if self.sort_reducers:
                key = self.order(r.LM)
                index = bisect.bisect(self.keysG_, key)
                self.G_.insert(index, r.monic())
                self.lmG_.insert(index, r.LM)
                self.keysG_.insert(index, key)
            else:
                self.G_ = self.G
                self.lmG_ = self.G_

        observation = (self.G, self.P)
        if self.mode == 'train':
            observation = make_obs(*observation)

        reward = -(1.0 + stats['steps']) if self.rewards == 'additions' else -1.0
        terminated = len(self.P) == 0
        truncated = False
        info = {'steps': stats['steps'], 'reduction_was_zero': r == 0, 'invalid_action': False}

        return observation, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.ideal_gen.seed(seed)

    def value(self, gamma=0.99):
        _, stats = buchberger([g.copy() for g in self.G],
                              S=self.P.copy(),
                              elimination=self.elimination,
                              rewards=self.rewards,
                              sort_reducers=self.sort_reducers,
                              gamma=gamma)
        return stats['discounted_return']

    def _make_ideal_gen(self, ideal_dist):
        """Return the ideal generator for this environment."""
        if isinstance(ideal_dist, IdealGenerator):
            return ideal_dist
        else:
            return parse_ideal_dist(ideal_dist)


def lead_monomials_vector(f, ring, k=2, dtype=np.int32):
    """Return the concatenated exponent vectors of the k lead monomials of f."""
    it = iter(f.monoms())
    return np.array([next(it, (0,) * ring.ngens) for _ in range(k)]).flatten().astype(dtype)


class LeadMonomialsEnv:
    """A BuchbergerEnv with state the matrix of the pairs' lead monomials.

    Parameters
    ----------
    ideal_dist : str, optional
        IdealGenerator or string naming the ideal distribution.
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.
    rewards : {'additions', 'reductions'}, optional
        Reward value for each step.
    sort_input : bool, optional
        Whether to sort the initial generating set by lead monomial.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order by lead monomial.
    k : int, optional
        Number of lead monomials shown for each polynomial.
    dtype : data-type, optional
        Data-type for the state matrix.

    Examples
    --------
    >>> env = LeadMonomialsEnv()
    >>> env.seed(123)
    >>> env.reset()
    array([[ 6,  4,  2,  0, 16,  3],
           [ 6,  4,  2, 18,  0,  2],
           [ 6,  4,  2, 11,  0,  8],
           [18,  0,  2, 11,  0,  8],
           [11,  0,  8,  3,  0, 17],
           [ 6,  4,  2,  2,  4, 13],
           [ 3,  0, 17,  2,  4, 13],
           [ 6,  4,  2,  6,  6,  5],
           [ 6,  4,  2,  2,  8,  6],
           [ 0, 16,  3,  2,  8,  6],
           [ 2,  4, 13,  2,  8,  6],
           [ 2,  8,  6,  4, 10,  6],
           [ 6,  4,  2,  2, 17,  0],
           [ 0, 16,  3,  2, 17,  0]], dtype=int32)
    >>> env.step(3)
    (array([[ 6,  4,  2,  0, 16,  3],
            [ 6,  4,  2, 18,  0,  2],
            [ 6,  4,  2, 11,  0,  8],
            [11,  0,  8,  3,  0, 17],
            [ 6,  4,  2,  2,  4, 13],
            [ 3,  0, 17,  2,  4, 13],
            [ 6,  4,  2,  6,  6,  5],
            [ 6,  4,  2,  2,  8,  6],
            [ 0, 16,  3,  2,  8,  6],
            [ 2,  4, 13,  2,  8,  6],
            [ 2,  8,  6,  4, 10,  6],
            [ 6,  4,  2,  2, 17,  0],
            [ 0, 16,  3,  2, 17,  0],
            [ 6,  4,  2,  4,  6, 10],
            [ 2,  4, 13,  4,  6, 10],
            [ 2,  8,  6,  4,  6, 10]], dtype=int32),
     -3.0,
     False,
     {})

    """

    def __init__(self, ideal_dist='3-20-10-uniform', elimination='gebauermoeller',
                 rewards='additions', sort_input=False, sort_reducers=True,
                 k=1, dtype=np.int32):
        self.env = BuchbergerEnv(ideal_dist, elimination, rewards, sort_input, sort_reducers)
        self.ring = self.env.ideal_gen.ring
        self.k = k
        self.dtype = dtype
        self.leads = []

    def reset(self):
        (G, _), _ = self.env.reset()
        self.leads = [lead_monomials_vector(g, self.ring, k=self.k, dtype=self.dtype) for g in G]

        return self._matrix(), {}

    def step(self, action):
        (G, P), reward, terminated, truncated, info = self.env.step(self.env.P[action])

        if len(G) > len(self.leads):
            self.leads.append(lead_monomials_vector(G[-1], self.ring, k=self.k, dtype=self.dtype))

        return self._matrix(), reward, terminated, truncated, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def value(self, gamma=0.99):
        return self.env.value(gamma)

    def _matrix(self):
        n = self.env.G[0].ring.ngens
        mat = np.empty((len(self.env.P), 2 * n * self.k), dtype=self.dtype)

        for i, p in enumerate(self.env.P):
            mat[i, :n*self.k] = self.leads[p[0]]
            mat[i, n*self.k:] = self.leads[p[1]]

        return mat


class LeadMonomialsAgent:
    """An agent that follows standard selection strategies.

    Parameters
    ----------
    selection : {'first', 'degree', 'random'}
        The selection strategy used to pick pairs.

    """

    def __init__(self, selection='degree', k=1):
        self.strategy = selection
        self.k = k

    def act(self, state):
        if self.strategy == 'first':
            return 0
        elif self.strategy == 'degree':
            n = state.shape[1] // (2 * self.k)
            m = state.shape[1] // 2
            return np.argmin(np.sum(np.maximum(state[:, :n], state[:, m:m+n]), axis=1))
        elif self.strategy == 'random':
            return np.random.choice(len(state))


class MCTSAgent:
    """
    An agent that uses MCTS (Monte Carlo Tree Search) to optimize pair selection.
    
    Parameters
    ----------
    env : BuchbergerEnv
        The Buchberger environment to use for simulations.
    n_simulations : int, optional
        Number of MCTS simulations to run per action selection.
    c : float, optional
        Exploration constant for UCB1 formula.
    gamma : float, optional
        Discount factor for future rewards.
    rollout_policy : str, optional
        Policy to use for rollouts ('random', 'normal', 'degree', 'first').
    """

    def __init__(self, env, n_simulations=50, c=1.0, gamma=0.99, rollout_policy='normal'):
        self.env = env
        self.n_simulations = n_simulations
        self.c = c
        self.gamma = gamma
        self.rollout_agent = BuchbergerAgent(selection=rollout_policy)

    def act(self, state):
        """
        Select the best action using MCTS.
        
        Parameters
        ----------
        state : tuple
            The current state (G, P) where G is the polynomial list and P is the pair set.
            
        Returns
        -------
        tuple
            The selected pair (i, j) to reduce.
        """
        G, P = state
        
        if len(P) == 0:
            return None
        
        if len(P) == 1:
            return P[0]
        
        # Initialize root node and run simulations
        root = MCTSNode(state=state, parent=None, action=None)
        
        for _ in range(self.n_simulations):
            self._run_simulation(root, G, P)
        
        # Return the action with the highest visit count
        return self._select_best_action(root)
    
    def _run_simulation(self, root, G, P):
        """Run a single MCTS simulation from the root node."""
        sim_env = self._copy_env_state(G, P)
        node = root
        
        # Selection phase
        node = self._select_node(node, sim_env)
        
        # Expansion phase
        node = self._expand_node(node, sim_env)
        
        # Simulation phase (rollout)
        total_reward = self._rollout(sim_env)
        
        # Backpropagation phase
        self._backpropagate(node, total_reward)
    
    def _select_node(self, node, sim_env):
        """
        Selection phase: traverse tree using UCB1.
        
        Parameters
        ----------
        node : MCTSNode
            The current node to start selection from.
        sim_env : BuchbergerEnv
            The simulation environment.
            
        Returns
        -------
        MCTSNode
            The selected node for expansion.
        """
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(self.c)
            _, _, terminated, truncated, _ = sim_env.step(node.action)
            if terminated or truncated:
                break
        return node
    
    def _expand_node(self, node, sim_env):
        """
        Expansion phase: add a new child node.
        
        Parameters
        ----------
        node : MCTSNode
            The node to expand from.
        sim_env : BuchbergerEnv
            The simulation environment.
            
        Returns
        -------
        MCTSNode
            The newly created child node, or the original node if terminal.
        """
        if not node.is_terminal():
            action = node.get_untried_action()
            if action is not None:
                obs, reward, terminated, truncated, info = sim_env.step(action)
                child_state = obs if not (terminated or truncated) else None
                node = node.add_child(action, child_state)
        return node
    
    def _rollout(self, sim_env):
        """
        Simulation phase: rollout to terminal state using the rollout policy.
        
        Parameters
        ----------
        sim_env : BuchbergerEnv
            The simulation environment.
            
        Returns
        -------
        float
            The total discounted reward from the rollout.
        """
        total_reward = 0.0
        discount = 1.0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if len(sim_env.P) == 0:
                break
            
            # Use BuchbergerAgent for rollout policy
            state = (sim_env.G, sim_env.P)
            action = self.rollout_agent.act(state)
            
            _, reward, terminated, truncated, _ = sim_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
        
        return total_reward
    
    def _backpropagate(self, node, total_reward):
        """
        Backpropagation phase: update node statistics up the tree.
        
        Parameters
        ----------
        node : MCTSNode
            The node to start backpropagation from.
        total_reward : float
            The total reward to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.value += total_reward
            node = node.parent
    
    def _select_best_action(self, root):
        """
        Select the best action from the root node.
        
        Parameters
        ----------
        root : MCTSNode
            The root node of the MCTS tree.
            
        Returns
        -------
        tuple
            The best action based on visit counts.
        """
        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def _copy_env_state(self, G, P):
        """Create a copy of the environment with the current state."""
        # Create a new environment instance
        env_copy = BuchbergerEnv(
            ideal_dist=self.env.ideal_gen,
            elimination=self.env.elimination,
            rewards=self.env.rewards,
            sort_input=self.env.sort_input,
            sort_reducers=self.env.sort_reducers,
            mode=self.env.mode
        )
        
        # Copy the state
        env_copy.G = [g.copy() for g in G]
        env_copy.lmG = [g.LM for g in env_copy.G]
        env_copy.P = P.copy()
        env_copy.order = G[0].ring.order if G else None
        
        if self.env.sort_reducers:
            env_copy.G_ = [g.copy() for g in self.env.G_]
            env_copy.lmG_ = [g.LM for g in env_copy.G_]
            env_copy.keysG_ = self.env.keysG_.copy()
        else:
            env_copy.G_ = env_copy.G
            env_copy.lmG_ = env_copy.lmG
        
        return env_copy


class MCTSNode:
    """
    A node in the MCTS tree.
    
    Parameters
    ----------
    state : tuple or None
        The state (G, P) at this node.
    parent : MCTSNode or None
        The parent node.
    action : tuple or None
        The action that led to this node.
    """
    
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(state[1]) if state is not None else []
    
    def is_fully_expanded(self):
        """Check if all actions from this node have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if this is a terminal node (no more pairs to reduce)."""
        return self.state is None or len(self.state[1]) == 0
    
    def get_untried_action(self):
        """Get an untried action from this node."""
        if len(self.untried_actions) > 0:
            return self.untried_actions.pop(0)
        return None
    
    def add_child(self, action, state):
        """Add a child node for the given action and state."""
        child = MCTSNode(state=state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def best_child(self, c):
        """
        Select the best child using UCB1 formula.
        
        Parameters
        ----------
        c : float
            Exploration constant.
            
        Returns
        -------
        MCTSNode
            The child with the highest UCB1 value.
        """
        return max(
            self.children,
            key=lambda child: (child.value / child.visits if child.visits > 0 else 0) +
                            c * np.sqrt(np.log(self.visits) / child.visits if child.visits > 0 else float('inf'))
        )
