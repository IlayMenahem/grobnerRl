import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import Module, filter_jit
from jaxtyping import Array

from grobnerRl.types import Observation


class MonomialEmbedder(Module):
    linear: eqx.nn.Linear

    def __init__(self, monomial_dim: int, embedding_dim: int, key: Array):
        self.linear = eqx.nn.Linear(monomial_dim, embedding_dim, key=key)

    @filter_jit
    def __call__(self, monomials: Array) -> Array:
        """
        Args:
        - monomials: Array of shape (num_monomials, monomial_dim)

        Returns:
        - Array of shape (num_monomials, embedding_dim)
        """
        x = jax.vmap(self.linear)(monomials)
        x = jax.nn.relu(x)

        return x


class PolynomialEmbedder(Module):
    """
    Polynomial embedder using DeepSets architecture in Equinox.
    """

    phi_layers: list[eqx.nn.Linear]
    rho_layers: list[eqx.nn.Linear]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        key: Array,
    ):
        keys = jax.random.split(key, hidden_layers * 2 + 2)
        self.phi_layers = [
            eqx.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, key=keys[i])
            for i in range(hidden_layers)
        ]
        self.rho_layers = [
            eqx.nn.Linear(
                hidden_dim if i == 0 else hidden_dim,
                output_dim if i == hidden_layers - 1 else hidden_dim,
                key=keys[hidden_layers + i],
            )
            for i in range(hidden_layers)
        ]

    @filter_jit
    def __call__(self, polynomial: Array, mask: Array | None = None) -> Array:
        """
        Args:
        - polynomial: Array of shape (num_monomials, input_dim)
        - mask: Optional boolean array of shape (num_monomials,) indicating valid monomials.

        Returns:
        - Array of shape (num_polynomials, output_dim)
        """
        h = polynomial
        for layer in self.phi_layers:
            h = jax.nn.relu(jax.vmap(layer)(h))
        
        if mask is not None:
            # Broadcast mask to (num_monomials, hidden_dim)
            # Use a large negative number for masked values so max pooling ignores them
            h = jnp.where(mask[:, None], h, -jnp.inf)

        h_pooled = jnp.max(h, axis=0)
        for layer in self.rho_layers:
            h_pooled = jax.nn.relu(layer(h_pooled))
        return h_pooled


class TransformerEncoderLayer(Module):
    attention: eqx.nn.MultiheadAttention
    layer_norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    layer_norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        key: Array,
    ):
        key_attn, key_mlp = jax.random.split(key, 2)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key_attn,
        )
        self.layer_norm1 = eqx.nn.LayerNorm(embedding_dim)
        self.mlp = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=embedding_dim,
            width_size=feedforward_dim,
            depth=1,
            activation=jax.nn.relu,
            key=key_mlp,
        )
        self.layer_norm2 = eqx.nn.LayerNorm(embedding_dim)

    @filter_jit
    def __call__(self, x: Array) -> Array:
        # x: (seq_len, embedding_dim)

        # Self-attention
        attn_out = self.attention(x, x, x)

        x = x + attn_out
        x = jax.vmap(self.layer_norm1)(x)

        mlp_out = jax.vmap(self.mlp)(x)
        x = x + mlp_out
        x = jax.vmap(self.layer_norm2)(x)

        return x


class IdealModel(Module):
    layers: list[TransformerEncoderLayer]

    def __init__(self, embedding_dim: int, num_heads: int, depth: int, key: Array):
        """
        Args:
        - embedding_dim: Dimension of polynomial embeddings
        - num_heads: Number of attention heads
        - depth: Number of transformer layers
        - key: JAX random key for initialization
        """
        keys = jax.random.split(key, depth)
        self.layers = [
            TransformerEncoderLayer(
                embedding_dim, num_heads, 4 * embedding_dim, keys[i]
            )
            for i in range(depth)
        ]

    @filter_jit
    def __call__(self, ideal_embeddings: Array) -> Array:
        """
        Args:
        - ideal_embeddings: Array of shape (num_polynomials, embedding_dim)

        Returns:
        - Array of shape (num_polynomials, embedding_dim)
        """
        x = ideal_embeddings
        for layer in self.layers:
            x = layer(x)
        return x


class PairwiseScorer(Module):
    mlp: eqx.nn.MLP

    def __init__(self, embedding_dim: int, hidden_dim: int, key: Array):
        self.mlp = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=key,
        )

    @filter_jit
    def __call__(self, embeddings: Array) -> Array:
        """
        Args:
        - embeddings: Array of shape (num_polynomials, embedding_dim)

        Returns:
        - Array of shape (num_polynomials, num_polynomials)
        """
        # Create pairwise combinations (symmetric)
        # (N, 1, D) + (1, N, D) -> (N, N, D)
        pairwise_emb = embeddings[:, None, :] + embeddings[None, :, :]

        # Apply MLP
        # We can use vmap to apply MLP over the N, N dimensions
        scores = jax.vmap(jax.vmap(self.mlp))(pairwise_emb)

        # scores is (N, N, 1), squeeze to (N, N)
        return jnp.squeeze(scores, axis=-1)


class Extractor(Module):
    monomial_embedder: MonomialEmbedder
    polynomial_embedder: PolynomialEmbedder
    ideal_model: IdealModel
    pairwise_scorer: PairwiseScorer

    def __init__(
        self,
        monomial_embedder: MonomialEmbedder,
        polynomial_embedder: PolynomialEmbedder,
        ideal_model: IdealModel,
        pairwise_scorer: PairwiseScorer,
    ):
        self.monomial_embedder = monomial_embedder
        self.polynomial_embedder = polynomial_embedder
        self.ideal_model = ideal_model
        self.pairwise_scorer = pairwise_scorer

    def __call__(self, obs: Observation) -> Array:
        ideal, selectables = obs

        # Pad polynomials to the same length
        ideal_arrays = [jnp.asarray(p) for p in ideal]
        lengths = [p.shape[0] for p in ideal_arrays]
        max_len = max(lengths) if lengths else 0

        padded_ideal = []
        masks = []
        for p in ideal_arrays:
            l = p.shape[0]
            pad_len = max_len - l
            if pad_len > 0:
                p_padded = jnp.pad(p, ((0, pad_len), (0, 0)), constant_values=0)
                mask = jnp.concatenate(
                    [jnp.ones(l, dtype=bool), jnp.zeros(pad_len, dtype=bool)]
                )
            else:
                p_padded = p
                mask = jnp.ones(l, dtype=bool)
            padded_ideal.append(p_padded)
            masks.append(mask)

        ideal_stacked = jnp.stack(padded_ideal)
        masks_stacked = jnp.stack(masks)

        monomial_embs = jax.vmap(self.monomial_embedder)(ideal_stacked)
        ideal_embeddings = jax.vmap(self.polynomial_embedder)(
            monomial_embs, masks_stacked
        )

        ideal_embeddings = self.ideal_model(ideal_embeddings)

        values = self.pairwise_scorer(ideal_embeddings)

        mask = jnp.full(values.shape, -jnp.inf)
        if selectables:
            rows, cols = zip(*selectables)
            rows = jnp.array(rows)
            cols = jnp.array(cols)
            mask = mask.at[rows, cols].set(0.0)

        values = values + mask
        return values.flatten()


class GrobnerPolicy(Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def __call__(self, obs: Observation) -> Array:
        # Return logits for training, not probabilities
        vals = self.extractor(obs)
        return vals


class GrobnerValue(Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def __call__(self, obs: Observation) -> Array:
        """
        Return the raw values produced by the extractor. For Equinox models we
        expect the extractor to return a JAX array (or a list of arrays for a batch).
        """
        vals = self.extractor(obs)
        return vals


class GrobnerCritic(Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def __call__(self, obs: Observation) -> Array:
        """
        Return a scalar (or per-batch scalars) representing the critic estimate.
        If the extractor returns a list/tuple of arrays (batched), take the max
        of each entry. Otherwise take the max over the flattened values.
        """
        vals = self.extractor(obs)
        state_value = jnp.max(vals)

        return state_value


if __name__ == "__main__":
    # Configuration
    num_vars = 10
    num_monomials = 4
    monoms_embedding_dim = 64
    polys_embedding_dim = 128
    ideal_depth = 8
    ideal_num_heads = 8
    num_polynomials = 100

    # Create JAX random keys
    key = jax.random.key(0)
    key, k_monomial, k_polynomial, k_ideal, k_pairwise = jax.random.split(key, 5)

    # Build Equinox modules
    monomial_embedder = MonomialEmbedder(
        monomial_dim=num_vars + 1, embedding_dim=monoms_embedding_dim, key=k_monomial
    )
    polynomial_embedder = PolynomialEmbedder(
        input_dim=monoms_embedding_dim,
        hidden_dim=polys_embedding_dim,
        hidden_layers=2,
        output_dim=polys_embedding_dim,
        key=k_polynomial,
    )
    ideal_model = IdealModel(
        embedding_dim=polys_embedding_dim,
        num_heads=ideal_num_heads,
        depth=ideal_depth,
        key=k_ideal,
    )
    pairwise_scorer = PairwiseScorer(
        embedding_dim=polys_embedding_dim,
        hidden_dim=polys_embedding_dim,
        key=k_pairwise,
    )

    # Compose extractor and high-level models
    extractor_eqx = Extractor(
        monomial_embedder=monomial_embedder,
        polynomial_embedder=polynomial_embedder,
        ideal_model=ideal_model,
        pairwise_scorer=pairwise_scorer,
    )

    policy_eqx = GrobnerPolicy(extractor_eqx)
    critic_eqx = GrobnerCritic(extractor_eqx)

    # Create a synthetic ideal: a list of polynomials, each a (num_monomials, num_vars+1) array
    key = jax.random.key(1)
    keys = jax.random.split(key, num_polynomials)
    ideal = [jax.random.normal(k, (num_monomials, num_vars + 1)) for k in keys]

    # Example selectables (allowed (row, col) pairs in the flattened pairwise matrix)
    selectables = [
        (i, j) for i in range(num_polynomials) for j in range(i + 1, num_polynomials)
    ]

    obs_eqx = (ideal, selectables)

    # Run policy and critic (Equinox models work with JAX arrays)
    policy_out = policy_eqx(obs_eqx)
    critic_out = critic_eqx(obs_eqx)

    print("Equinox Policy output shape:", policy_out.shape)
    print("Equinox Critic output:", critic_out)
