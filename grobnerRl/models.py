from typing import Sequence
import equinox as eqx
import jax
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array
from sympy.polys.rings import PolyElement

from grobnerRl.rl.utils import GroebnerState


class EmbeddingMonomials(eqx.Module):
    embedding: eqx.nn.Embedding

    def __init__(self, num_embeddings: int, output_dim: int, key):
        '''
        Args:
        num_embeddings: int - The number of embeddings, this is also the number of different
        variables that the model can handle.
        output_dim: int - The dimension of the output space
        key - The key for random initialization
        '''
        self.embedding = eqx.nn.Embedding(num_embeddings, output_dim, key=key)

    def __call__(self, monomials: Array) -> Array:
        '''
        embeds a monomials into a vector space of dimension output_dim

        Args:
        monomials: Array (num_monomials, num_vars) - The monomials to be embedded

        Returns:
        Array (num_monomials, output_dim) - The embedded monomials
        '''
        num_vars = monomials.shape[-1]
        embeddings = vmap(self.embedding)(jnp.arange(num_vars))
        embedded = jnp.matmul(monomials, embeddings)

        return embedded


class Transformer(eqx.Module):
    layers: list[tuple[eqx.nn.MultiheadAttention, eqx.nn.Linear]]
    linear: eqx.nn.Linear

    def __init__(self, input_dim: int, depth: int, num_heads: int, key):
        '''
        Args:
        input_dim: int - The dimension of the input
        depth: int - The depth of the transformer
        num_heads: int - The number of attention heads in each multi-head attention layer
        key: PRNGKey - The key for random initialization

        Returns:
        Transformer - The initialized transformer
        '''
        keys = jax.random.split(key, 2*depth+1)

        self.layers = [
            (eqx.nn.MultiheadAttention(num_heads, input_dim, key=keys[2*i]), eqx.nn.Linear(input_dim, input_dim, key=keys[2*i+1]))
            for i in range(depth)
        ]
        self.linear = eqx.nn.Linear(input_dim, input_dim, key=keys[-1])

    def __call__(self, x: Array) -> Array:
        '''
        Args:
        x: Array (seq_len, input_dim) - a 2d array of shape (seq_len, input_dim)
        key: PRNGKey - The key for random initialization

        Returns:
        Array (seq_len, input_dim) - The embedded input array
        '''
        input = x

        for attention, linear in self.layers:
            attention_output = attention(input, input, input, key=None) + input
            output = jax.nn.gelu(vmap(linear)(attention_output)) + attention_output

            input = output

        output = vmap(self.linear)(input)

        return output


class TransformerEmbedder(eqx.Module):
    transformer: Transformer
    adaptor: eqx.nn.Linear

    def __init__(self, input_dim: int, output_dim: int, depth: int, num_heads: int, key):
        '''
        Args:
        input_dim: int - The dimension of the input
        output_dim: int - The dimension of the output
        depth: int - The depth of the transformer
        num_heads: int - The number of heads in the transformer
        key: PRNGKey - The key for random initialization
        '''
        key1, key2 = jax.random.split(key, 2)

        self.transformer = Transformer(input_dim, depth, num_heads, key=key1)
        self.adaptor = eqx.nn.Linear(input_dim, output_dim, key=key2)

    def __call__(self, x: Array) -> Array:
        '''
        Args:
        x: Array (seq_len, input_dim) - a 2d array of shape (seq_len, input_dim)

        Returns:
        Array (output_dim) - The embedded input array
        '''
        x = self.transformer(x)
        x = jnp.mean(x, axis=0)
        x = self.adaptor(x)

        return x


def tokenize(ideal: Sequence[PolyElement]) -> Array:
    '''
    takes an ideal and returns a tokenized version of it, a list of arrays, each of the arrays
    representing a polynomial monomials

    Parameters:
    ideal: list[PolyElement] - The ideal generators to be tokenized

    Returns:
    tokenized ideal
    '''
    polys_monomials = [jnp.array(poly.monoms()) for poly in ideal]

    max_len = max(len(mono) for mono in polys_monomials)
    padded_monos = [jnp.pad(mono, ((0, max_len - len(mono)), (0, 0))) for mono in polys_monomials]
    tokenized_ideal = jnp.stack(padded_monos)

    return tokenized_ideal


def make_obs(G, P):
    G = tokenize(G)
    P = jnp.array(P)
    obs = GroebnerState(G, P)
    return obs


class GrobnerExtractor(eqx.Module):
    monomial_model: EmbeddingMonomials
    polynomial_model: TransformerEmbedder
    ideal_model: Transformer

    def __init__(self, vars_limit: int, monoms_embedding_dim: int, polys_embedding_dim: int, polys_depth: int,
        polys_num_heads: int, ideal_depth: int, ideal_num_heads: int, key):
        '''
        Args:
        - vars_limit: int - The maximum number of variables in the polynomial ring
        - monoms_embedding_dim: int - The dimension of the embedding space for monomials
        - polys_embedding_dim: int - The dimension of the embedding space for polynomials
        - polys_depth: int - The depth of the transformer model for polynomials
        - polys_num_heads: int - The number of attention heads in the transformer model for polynomials
        - ideal_depth: int - The depth of the transformer model for ideals
        - ideal_num_heads: int - The number of attention heads in the transformer model for ideals
        - key: jax.random.PRNGKey - The random key for initialization
        '''
        key1, key2, key3 = jax.random.split(key, 3)

        self.monomial_model = EmbeddingMonomials(vars_limit, monoms_embedding_dim, key1)
        self.polynomial_model = TransformerEmbedder(monoms_embedding_dim, polys_embedding_dim, polys_depth, polys_num_heads, key2)
        self.ideal_model = Transformer(polys_embedding_dim, ideal_depth, ideal_num_heads, key3)

    @eqx.filter_jit
    def __call__(self, ideal) -> Array:
        '''
        scores each pair of polynomials to select to reduce in buchberger's algorithm

        Args:
        - ideal: Array - The ideal generators

        Returns:
        2d Array of scores for selecting a polynomial pair
        '''
        monomial_embeddings: Array = vmap(self.monomial_model)(ideal)
        polynomial_embeddings: Array = vmap(self.polynomial_model)(monomial_embeddings)

        polynomial_arrays = self.ideal_model(polynomial_embeddings)
        values = jnp.matmul(polynomial_arrays, polynomial_arrays.T)

        return values


@eqx.filter_jit
def mask_selectables(values, selectables, masking_value):
    mask = jnp.zeros_like(values)
    mask = mask.at[tuple(zip(*selectables))].set(1)
    values = jnp.where(mask == 1, values, masking_value)
    return values


class GrobnerPolicy(eqx.Module):
    model: GrobnerExtractor

    def __init__(self, groebner_model: GrobnerExtractor):
        self.model = groebner_model

    def __call__(self, obs: GroebnerState) -> Array:
        vals = self.model(obs.ideal)
        vals = mask_selectables(vals, obs.selectables, -jnp.inf)
        probs = jax.nn.softmax(vals, axis=None)

        return probs


class GrobnerCritic(eqx.Module):
    model: GrobnerExtractor

    def __init__(self, groebner_model: GrobnerExtractor):
        self.model = groebner_model

    def __call__(self, obs: GroebnerState) -> Array:
        vals = self.model(obs.ideal)
        value = jnp.mean(vals)

        return value
