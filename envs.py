from sympy.polys.rings import PolyElement, xring
from sympy.polys.domains import ZZ
from sympy.polys.orderings import lex
from jaxtyping import Array, Bool

def generate_monomial():
    raise NotImplementedError("generate_monomial is not implemented yet.")

def generate_polynomial():
    raise NotImplementedError("generate_polynomial is not implemented yet.")

def generate_ideal(num_poly: int, vars: list[str]) -> list[PolyElement]:
    return [generate_polynomial() for _ in range(num_poly)]

class GrobnerEnv:
    generators: list[PolyElement]
    is_reduciable: Bool[Array, ...]
    field: str

    def __init__(self):
        # create the ring
        num_vars = 3
        vars_names = [f'x{i}' for i in range(1, num_vars + 1)]
        R, vars = xring(vars_names, ZZ, order=lex)

        # create a new random finitely generated ideal in sympy over the $Z[x_1, x_2, ..., x_n]$
        num_polynomials = 5
        ideal = generate_ideal(num_polynomials, vars)

        # create the is_reduciable array

        raise NotImplementedError("GrobnerEnv is not implemented yet.")

    def reset(self, seed=None):
        '''
        Initialize the environment
        '''
        raise NotImplementedError("GrobnerEnv.reset is not implemented yet.")

    def step(self, action: tuple[int, int]) -> tuple[float, list[PolyElement], bool, bool, dict]:
        '''
        does the action in the environment

        Parameters:
        - action (tuple[int, int]): The indecies of the polynomials to reduce

        Returns (tuple[float, list[PolyElement], bool, bool, dict]):
        - reward
        - observation tuple[list[PolyElement], jnp.ndarray]: the polynomials, and which tuples can be recuded
        - done
        - truncated
        - info
        '''
        raise NotImplementedError("GrobnerEnv.step is not implemented yet.")
