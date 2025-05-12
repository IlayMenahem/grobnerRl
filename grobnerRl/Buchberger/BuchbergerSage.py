from sage.structure.sequence import Sequence
from sage.all_cmdline import *

def LCM(f, g):
    return f.parent().monomial_lcm(f, g)

def LM(f):
    return f.lm()

def LT(f):
    return f.lt()


def spol(f, g):
    """
    Compute the S-polynomial of f and g.

    INPUT:

    - ``f``, ``g`` -- polynomials

    OUTPUT: the S-polynomial of f and g

    EXAMPLES::

        sage: R.<x,y,z> = PolynomialRing(QQ)
        sage: from sage.rings.polynomial.toy_buchberger import spol
        sage: spol(x^2 - z - 1, z^2 - y - 1)
        x^2*y - z^3 + x^2 - z^2
    """
    fg_lcm = LCM(LM(f), LM(g))
    return fg_lcm//LT(f)*f - fg_lcm//LT(g)*g


def buchberger(F):
    """
    Compute a Groebner basis using the original version of Buchberger's
    algorithm as presented in [BW1993]_, page 214.

    Parameters:
    - F: an ideal in a multivariate polynomial ring

    Returns: A Groebner basis for F
    """
    G = set(F.gens())
    B = set((g1, g2) for g1 in G for g2 in G if g1 != g2)

    while B:
        g1, g2 = select(B)
        B.remove((g1, g2))

        h = spol(g1, g2).reduce(G)
        if h != 0:
            B = B.union((g, h) for g in G)
            G.add(h)

    return Sequence(G)


def buchberger_improved(F):
    """
    Compute a Groebner basis using an improved version of Buchberger's
    algorithm as presented in [BW1993]_, page 232.

    This variant uses the Gebauer-Moeller Installation to apply
    Buchberger's first and second criterion to avoid useless pairs.

    Parameters:
    - F: an ideal

    Returns: A Groebner basis for F
    """
    F = inter_reduction(F.gens())

    G = set()
    B = set()

    while F:
        f = min(F)
        F.remove(f)
        G, B = update(G, B, f)

    while B:

        g1, g2 = select(B)
        B.remove((g1, g2))
        h = spol(g1, g2).reduce(G)
        if h != 0:
            G, B = update(G, B, h)

    return Sequence(inter_reduction(G))


def update(G, B, h):
    """
    Update ``G`` using the set of critical pairs ``B`` and the
    polynomial ``h`` as presented in [BW1993]_, page 230. For this,
    Buchberger's first and second criterion are tested.

    This function implements the Gebauer-Moeller Installation.

    Parameters:
    - ``G`` -- an intermediate Groebner basis
    - ``B`` -- set of critical pairs
    - ``h`` -- a polynomial

    Returns: a tuple of
    - an intermediate Groebner basis
    - a set of critical pairs
    """
    R = h.parent()

    C = set((h, g) for g in G)
    D = set()

    while C:
        (h, g) = C.pop()

        def lcm_divides(rhs):
            return R.monomial_divides(LCM(LM(h), LM(rhs[1])),
                                      LCM(LM(h), LM(g)))

        if R.monomial_pairwise_prime(LM(h), LM(g)) or \
                (
                   not any(lcm_divides(f) for f in C)
                   and
                   not any(lcm_divides(f) for f in D)
                ):
            D.add((h, g))

    E = set()

    while D:
        (h, g) = D.pop()
        if not R.monomial_pairwise_prime(LM(h), LM(g)):
            E.add((h, g))

    B_new = set()

    while B:
        g1, g2 = B.pop()
        if not R.monomial_divides(LM(h), LCM(LM(g1), LM(g2))) or \
           R.monomial_lcm(LM(g1), LM(h)) == LCM(LM(g1), LM(g2)) or \
           R.monomial_lcm(LM(h), LM(g2)) == LCM(LM(g1), LM(g2)):
            B_new.add((g1, g2))

    B_new = B_new.union(E)

    G_new = set()

    while G:
        g = G.pop()
        if not R.monomial_divides(LM(h), LM(g)):
            G_new.add(g)

    G_new.add(h)

    return G_new, B_new


def select(P):
    """
    Select a polynomial using the normal selection strategy.

    Parameters:
    - P: list of critical pairs

    Returns: an element of P
    """
    return min(P, key=lambda fi_fj: LCM(LM(fi_fj[0]),
                                        LM(fi_fj[1])).total_degree())


def inter_reduction(Q):
    r"""
    Compute inter-reduced polynomials from a set of polynomials.

    Parameters:
    - Q: set of polynomials

    Returns: a set of inter-reduced polynomials
    if ``Q`` is the set `f_1, ..., f_n`, this method returns `g_1,
    ..., g_s` such that:

    - `(f_1,...,f_n) = (g_1,...,g_s)`
    - `LM(g_i) \neq LM(g_j)` for all `i \neq j`
    - `LM(g_i)` does not divide `m` for all monomials `m` of
      `\{g_1,...,g_{i-1}, g_{i+1},...,g_s\}`
    - `LC(g_i) = 1` for all `i`.
    """
    if not Q:
        return Q  # if Q is empty we cannot get a base ring
    base_ring = next(iter(Q)).base_ring()

    Q = set(Q)
    while True:
        Qbar = set(Q)
        for p in sorted(Qbar):
            Q.remove(p)
            h = p.reduce(Q)
            if not h.is_zero():
                Q.add(h)
        if Qbar == Q:
            if base_ring.is_field():
                return set(f.lc()**(-1) * f for f in Qbar)
            else:
                return Qbar


_sage_const_0 = Integer(0)
_sage_const_1 = Integer(1)

class SignedMatrix:
    # Matrix together with hashmap associating singature (index) to each row - has special rref function that respects signatures.
    def __init__(self, mat, sgn, d, parent):
        self.mat = mat
        self.signature = sgn
        self.d = d
        self.parent = parent

    # use position over term ordering
    def row_echelon_form_by_position(self):
        # returns a pair (M, n) where M is a new signed matrix which is the row-reduction of self via a sequence of
        # elementary row operations
        # keep track of number of operations
        num_operations = _sage_const_0
        copy_mat = copy(self.mat)
        eliminated = True
        first_reduction = True
        rdxn = dict()

        for i in range(len(copy_mat.rows())):
            rdxn[i] = []

        while eliminated:
            eliminated = False

            for i, row in enumerate(copy_mat.rows()):
                for j in range(len(row)):
                    if row[j] != _sage_const_0 :
                        # j is the leading term of this row, so use it to kill everything with higher signature
                        for new_i, new_row in enumerate(copy_mat.rows()):
                            if new_row[j] != _sage_const_0  and self.signature[i] < self.signature[new_i]:
                                # we can reduce
                                lam = -(new_row[j]/row[j])
                                copy_mat.add_multiple_of_row(new_i, i, lam)
                                eliminated = True
                                if first_reduction: # only count top-reductions
                                    num_operations += len(new_row)
                                rdxn[new_i].append((i,lam))
                        break
            first_reduction = False # stop counting arithmetic operations

        for i, row in enumerate(copy_mat.rows()):
            for j in range(len(row)):
                if row[j] != _sage_const_0 :
                    # j is the coefficient of the leading term of this row, so divide this row by it
                    copy_mat.rescale_row(i,_sage_const_1 /row[j])
                    break

        return (SignedMatrix(copy_mat, self.signature, self.d, self.parent), num_operations, rdxn)

    # use term over position ordering
    def row_echelon_form_by_term(self):
        num_operations = _sage_const_0
        copy_mat = copy(self.mat)
        eliminated = True
        first_reduction = True

        while eliminated:
            eliminated = False

            for i, row in enumerate(copy_mat.rows()):
                for j in range(len(row)):
                    if row[j] != _sage_const_0 :
                        # j is the leading term of this row, so use it to kill everything with higher signature
                        for new_i, new_row in enumerate(copy_mat.rows()):
                            if new_row[j] != _sage_const_0  and self.signature[i][::-_sage_const_1 ] < self.signature[new_i][::-_sage_const_1 ]:
                                # we can reduce
                                lam = -(new_row[j]/row[j])
                                copy_mat.add_multiple_of_row(new_i, i, lam)
                                eliminated = True
                                if first_reduction:
                                    num_operations += len(new_row)
                        break

            first_reduction = False

        for i, row in enumerate(copy_mat.rows()):
            for j in range(len(row)):
                if row[j] != _sage_const_0 :
                    # j is the coefficient of the leading term of this row, so divide this row by it
                    copy_mat.rescale_row(i,_sage_const_1 /row[j])
                    break

        return (SignedMatrix(copy_mat, self.signature, self.d, self.parent), num_operations)

    def add_row(self, f, index):
        # returns a new matrix which is self with a row added corresponding to polynomial f with signature index
        row = [f.monomial_coefficient(mon) for mon in self.monomials()]
        copy_mat = copy(self.mat)
        copy_signature = copy(self.signature)
        copy_mat = matrix(copy_mat.rows()+[row])
        copy_signature[copy_mat.nrows()-_sage_const_1 ] = index

        return SignedMatrix(copy_mat, copy_signature, self.d, self.parent)

    def monomials(self):
        # returns monomials of degree self.d in a list, sorted in decreasing order
        R = self.parent
        monomials = [R({tuple(a):_sage_const_1 }) for a in WeightedIntegerVectors(self.d, [_sage_const_1 ]*R.ngens())]
        monomials.sort(reverse=True)

        return monomials

    def LT(self):
        # returns the leading terms of the (polynomials represented by) rows of self.mat
        monomials = self.monomials()
        leading_terms = []

        for row in self.mat.rows():
            for i in range(len(row)):
                if row[i] != _sage_const_0 :
                    leading_terms.append(monomials[i]*row[i])
                    break

        return set(leading_terms)

    def rows(self):
        # return set of (polynomials represented by) rows of self.mat
        monomials = self.monomials()
        r = []
        for row in self.mat.rows():
            polynomial = _sage_const_0
            for j in range(len(row)):
                polynomial += row[j]*monomials[j]
            r.append(polynomial)
        return r

def F5(F, D, order='position'):
    # F=(f_1,...,f_m) is a set of polynomials with degere d_1 <= d_2 <= ... <= d_m
    # D is maximal degree
    # returns the set of elements of degree at most D of reduced Grobner bases of (f_1,...,f_i) for each i
    operations = _sage_const_0
    F.insert(_sage_const_0 ,_sage_const_0 ) # so that we can 1-index everything
    G = [{} for _ in range(len(F))] # initialize intermediate Grobner bases
    M = [[None for _ in range(len(F))] for _ in range(D+_sage_const_1 )] # initialize Macaulay matrices
    M_red = [[None for _ in range(len(F))] for _ in range(D+_sage_const_1 )] # initialize reduced Macaulay matrices
    sizes = [] # initialize list of sizes of Macaulay matrices
    rdxn = [[None for _ in range(len(F))] for _ in range(D+_sage_const_1 )] # initialize list of reductions performed
    variables = list(F[_sage_const_1 ].parent().gens())
    variables.sort(reverse=True)

    for d in range(F[_sage_const_1 ].degree(),D+_sage_const_1 ):
        M[d][_sage_const_0 ] = SignedMatrix(matrix(QQ), dict(), d, F[_sage_const_1 ].parent())
        M_red[d][_sage_const_0 ] = SignedMatrix(matrix(QQ), dict(), d, F[_sage_const_1 ].parent())

        for i in range(_sage_const_1 , len(F)):
            if d < F[i].degree():
                M[d][i] = M[d][i-_sage_const_1 ] # Case 1: the degree of f_i is larger than d
            elif d == F[i].degree(): # Case 2: the degree of f_i is exactly d
                M[d][i] = M_red[d][i-_sage_const_1 ].add_row(F[i], (i,_sage_const_1 ))
            else: # Case 3: the degree of f_i is less than d
                M[d][i] = M_red[d][i-_sage_const_1 ]
                if M_red[d-F[i].degree()][i-_sage_const_1 ]:
                    Crit = M_red[d-F[i].degree()][i-_sage_const_1 ].LT() # build F_5 criterion list
                else:
                    Crit = []
                for row_num, sgn in [(r,s) for (r,s) in M[d-_sage_const_1 ][i].signature.items() if s not in M[d-_sage_const_1 ][i-_sage_const_1 ].signature.values()]:
                    _,u = sgn
                    f = M[d-_sage_const_1 ][i].rows()[row_num]
                    if u == _sage_const_1 :
                        largest_var_in_u = _sage_const_0
                    else:
                        largest_var_in_u = variables.index(u.variables()[-_sage_const_1 ]) # select which row to use to build new row
                    for j in range(largest_var_in_u,len(variables)):
                        if u*variables[j] not in Crit: # avoid signatures which F_5 criterion tells us are useless
                            M[d][i] = M[d][i].add_row(variables[j]*f, (i,u*variables[j]))

            # reduce Macaulay-like matrices
            if order == "position":
                M_red[d][i], op, rdxn[d][i] = M[d][i].row_echelon_form_by_position()
                operations += op
            else:
                M_red[d][i], op, rdxn[d][i] = M[d][i].row_echelon_form_by_term()
                operations += op
            sizes.append((M[d][i].mat.nrows(), M[d][i].mat.ncols()))
            # update Grobner bases
            for j,f in enumerate(M_red[d][i].rows()):
                if f.lt().reduce([g.lt() for g in G[i].values()]) != _sage_const_0 :
                    G[i][M_red[d][i].signature[j]] = f

    return (G,sizes,M[-_sage_const_1 ][-_sage_const_1 ].mat,M[-_sage_const_1 ][-_sage_const_1 ].signature, M_red, operations, rdxn)
