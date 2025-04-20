import equinox as eqx
import jax
from jaxtyping import Array
from sympy.polys.rings import PolyElement

class EmbeddingVars(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, var):
        raise NotImplementedError

class EmbeddingMonomials(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, monomials):
        raise NotImplementedError

class EmbeddingPolynomials(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, polynomials):
        raise NotImplementedError

class IdealExtractor(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, ideals):
        raise NotImplementedError

class Evaluator(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, scores):
        raise NotImplementedError

def tokenize(ideal: list[PolyElement]) -> list[set[set[tuple[int, int]]]]:
    '''

    '''
    raise NotImplementedError

class GrobnerModel(eqx.Module):
    vars_embedder: EmbeddingVars
    monomial_model: EmbeddingMonomials
    polynomial_model: EmbeddingPolynomials
    ideal_extractor: IdealExtractor
    evaluator: Evaluator

    def __init__(self):
        pass

    def __call__(self, ideal: list[PolyElement]) -> Array:
        '''
        scores each pair of polynomials to select to reduce in buchberger's algorithm

        Args:
        ideal: list[PolyElement]

        Returns:
        2d Array of scores for selecting a polynomial pair
        '''
        ideal = tokenize(ideal)
        vars_embeddings: list[set[set[Array]]] = jax.vmap(self.vars_embedder)(ideal)
        monomial_embeddings: list[set[Array]] = jax.vmap(self.monomial_model)(vars_embeddings)
        polynomial_embeddings: list[Array] = jax.vmap(self.polynomial_model)(monomial_embeddings)
        ideal_embeddings: Array = jax.vmap(self.ideal_extractor)(polynomial_embeddings)
        scores: Array = self.evaluator(ideal_embeddings)

        return scores
