import jax.numpy as jnp
import chex
from grobnerRl.models import tokenize
from sympy.polys.orderings import lex
from sympy.polys.rings import ring
from sympy.polys.domains import QQ

def test_make_obs():
    R, x,y = ring("x,y", QQ, lex)
    f = x**2 + 2*x*y**2
    g = x*y + 2*y**3 - 1
    ideal = [f, g]

    expected = jnp.array([[[2, 0],
                         [1, 2],
                         [0, 0]],

                        [[1, 1],
                         [0, 3],
                         [0, 0]]])


    chex.assert_trees_all_equal(tokenize(ideal), expected)
