import numpy as np
from grobnerRl.envs.deepgroebner import tokenize
from sympy.polys.orderings import lex
from sympy.polys.rings import ring
from sympy.polys.domains import QQ

def test_make_obs():
    R, x,y = ring("x,y", QQ, lex)
    f = x**2 + 2*x*y**2
    g = x*y + 3*y**3 - 1
    ideal = [f, g]

    expected = [np.array([[1, 2, 0],
                         [2, 1, 2]]),

                np.array([[1, 1, 1],
                         [3, 0, 3],
                         [-1, 0, 0]])]

    obs = tokenize(ideal)

    np.testing.assert_array_equal(obs[0], expected[0])
    np.testing.assert_array_equal(obs[1], expected[1])
