from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module, filter_jit
from jaxtyping import Array

from grobnerRl.types import Observation


def _next_pow2(n: int) -> int:
    """Round n up to the next power of two (with a floor of 1)."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class RelationalLayer(eqx.Module):
    """
    A single relational message-passing layer for sets of polynomials.
    Instead of self-attention, this computes explicit pairwise messages
    between all polynomials and aggregates them.
    """

    message_mlp: eqx.nn.MLP
    update_mlp: eqx.nn.MLP
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, key: Array):
        m_key, u_key = jax.random.split(key, 2)

        # Message function takes pair of embeddings
        self.message_mlp = eqx.nn.MLP(
            in_size=embedding_dim * 2,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=m_key,
        )

        # Update function takes original embedding + aggregated message
        self.update_mlp = eqx.nn.MLP(
            in_size=embedding_dim + hidden_dim,
            out_size=embedding_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=u_key,
        )

        self.layer_norm = eqx.nn.LayerNorm(embedding_dim)

    @eqx.filter_jit
    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        """
        Args:
            x: Array of shape (num_polynomials, embedding_dim)
            mask: Optional boolean array of shape (num_polynomials,)

        Returns:
            Updated array of shape (num_polynomials, embedding_dim)
        """
        num_polys, dim = x.shape

        # Pre-LayerNorm to keep training stable
        normed_x = jax.vmap(self.layer_norm)(x)

        # Create all pairs: (N, 1, D) and (1, N, D) -> (N, N, 2D)
        x_i = jnp.broadcast_to(normed_x[:, None, :], (num_polys, num_polys, dim))
        x_j = jnp.broadcast_to(normed_x[None, :, :], (num_polys, num_polys, dim))
        pairs = jnp.concatenate([x_i, x_j], axis=-1)

        # Compute messages (N, N, hidden_dim)
        messages = jax.vmap(jax.vmap(self.message_mlp))(pairs)

        # Apply mask to mask out messages from invalid senders
        if mask is not None:
            # Mask over the sender dimension (axis 1)
            msg_mask = mask[None, :, None]
            messages = jnp.where(msg_mask, messages, 0.0)

            num_valid = jnp.sum(mask) + 1e-9
            agg_messages = jnp.sum(messages, axis=1) / num_valid
        else:
            agg_messages = jnp.mean(messages, axis=1)

        # Update step (N, D + hidden_dim)
        update_input = jnp.concatenate([normed_x, agg_messages], axis=-1)
        update_delta = jax.vmap(self.update_mlp)(update_input)

        # Residual connection
        new_x = x + update_delta

        if mask is not None:
            new_x = jnp.where(mask[:, None], new_x, 0.0)

        return new_x


class RelationalIdealModel(eqx.Module):
    """
    Simpler alternative to the Transformer-based IdealModel.
    Uses Relational Networks (Message Passing) across the unordered set
    of polynomials to build contextual embeddings.
    """

    layers: list[RelationalLayer]

    def __init__(self, embedding_dim: int, hidden_dim: int, depth: int, key: Array):
        """
        Args:
            embedding_dim: Dimension of polynomial embeddings
            hidden_dim: Hidden size for message passing networks
            depth: Number of relational layers
            key: JAX random key for initialization
        """
        keys = jax.random.split(key, depth)
        self.layers = [
            RelationalLayer(embedding_dim, hidden_dim, keys[i]) for i in range(depth)
        ]

    @eqx.filter_jit
    def __call__(self, ideal_embeddings: Array, mask: Array | None = None) -> Array:
        """
        Args:
            ideal_embeddings: Array of shape (num_polynomials, embedding_dim)
            mask: Optional boolean array of shape (num_polynomials,)

        Returns:
            Array of shape (num_polynomials, embedding_dim)
        """
        x = ideal_embeddings
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


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
    DeepSets polynomial embedder with LayerNorm and an explicit leading-monomial
    index. The final rho layer is linear (no ReLU) so the polynomial embedding
    can take negative values.
    """

    phi_layers: list[eqx.nn.Linear]
    rho_layers: list[eqx.nn.Linear]
    phi_norm: eqx.nn.LayerNorm
    pool_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        key: Array,
    ):
        keys = jax.random.split(key, hidden_layers * 2)
        self.phi_layers = [
            eqx.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, key=keys[i])
            for i in range(hidden_layers)
        ]
        self.rho_layers = [
            eqx.nn.Linear(
                hidden_dim,
                output_dim if i == hidden_layers - 1 else hidden_dim,
                key=keys[hidden_layers + i],
            )
            for i in range(hidden_layers)
        ]
        self.phi_norm = eqx.nn.LayerNorm(hidden_dim)
        self.pool_norm = eqx.nn.LayerNorm(hidden_dim)

    @filter_jit
    def __call__(
        self,
        polynomial: Array,
        mask: Array | None = None,
        lm_index: int = 0,
    ) -> Array:
        """
        Args:
        - polynomial: Array of shape (num_monomials, input_dim)
        - mask: Optional boolean array of shape (num_monomials,) indicating valid monomials.
        - lm_index: index of the leading monomial in `polynomial` (default 0,
          which is the convention used by `tokenize` for sympy's grevlex order).

        Returns:
        - Array of shape (output_dim,)
        """
        h = polynomial
        for layer in self.phi_layers:
            h = jax.nn.relu(jax.vmap(layer)(h))
        h = jax.vmap(self.phi_norm)(h)

        lm_emb = h[lm_index]
        if mask is not None:
            h = jnp.where(mask[:, None], h, -jnp.inf)

        h_pooled = jnp.max(h, axis=0)
        h_pooled = jnp.where(jnp.isneginf(h_pooled), 0.0, h_pooled)
        x = self.pool_norm(lm_emb + h_pooled)

        for layer in self.rho_layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.rho_layers[-1](x)

        return x


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
    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        # x: (seq_len, embedding_dim)

        if mask is not None:
            # Symmetric mask: zero attention to and from invalid positions.
            attn_mask = mask[None, :] & mask[:, None]
        else:
            attn_mask = None

        x_norm1 = jax.vmap(self.layer_norm1)(x)
        attn_out = self.attention(x_norm1, x_norm1, x_norm1, mask=attn_mask)
        if mask is not None:
            attn_out = jnp.where(mask[:, None], attn_out, 0.0)

        x = x + attn_out

        x_norm2 = jax.vmap(self.layer_norm2)(x)
        mlp_out = jax.vmap(self.mlp)(x_norm2)
        if mask is not None:
            mlp_out = jnp.where(mask[:, None], mlp_out, 0.0)
        x = x + mlp_out

        if mask is not None:
            x = jnp.where(mask[:, None], x, 0.0)

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
    def __call__(self, ideal_embeddings: Array, mask: Array | None = None) -> Array:
        """
        Args:
        - ideal_embeddings: Array of shape (num_polynomials, embedding_dim)
        - mask: Optional boolean array of shape (num_polynomials,)

        Returns:
        - Array of shape (num_polynomials, embedding_dim)
        """
        x = ideal_embeddings
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class PairwiseScorer(Module):
    """
    Symmetric bilinear scorer: score(i, j) = p_i^T W_sym p_j + b,
    where W_sym = (W + W^T) / 2. Permutation-equivariant in the polynomial
    set and symmetric in (i, j), so score(i, j) == score(j, i) by construction.

    The `hidden_dim` argument is accepted for backward compatibility with the
    previous MLP-based scorer; it is unused.
    """

    W: Array
    bias: Array

    def __init__(self, embedding_dim: int, hidden_dim: int, key: Array):
        del hidden_dim
        scale = 1.0 / jnp.sqrt(embedding_dim)
        self.W = scale * jax.random.normal(key, (embedding_dim, embedding_dim))
        self.bias = jnp.zeros(())

    @filter_jit
    def __call__(self, embeddings: Array) -> Array:
        """
        Args:
        - embeddings: Array of shape (num_polynomials, embedding_dim)

        Returns:
        - Array of shape (num_polynomials, num_polynomials), symmetric.
        """
        w_sym = 0.5 * (self.W + self.W.T)
        scores = embeddings @ w_sym @ embeddings.T
        return scores + self.bias


class Extractor(Module):
    monomial_embedder: MonomialEmbedder
    polynomial_embedder: PolynomialEmbedder
    ideal_model: IdealModel | RelationalIdealModel
    pairwise_scorer: PairwiseScorer

    def __init__(
        self,
        monomial_embedder: MonomialEmbedder,
        polynomial_embedder: PolynomialEmbedder,
        ideal_model: IdealModel | RelationalIdealModel,
        pairwise_scorer: PairwiseScorer,
    ):
        self.monomial_embedder = monomial_embedder
        self.polynomial_embedder = polynomial_embedder
        self.ideal_model = ideal_model
        self.pairwise_scorer = pairwise_scorer

    def __call__(self, obs: Observation | dict) -> Array:
        if isinstance(obs, dict):
            ideal_stacked = obs["ideals"]
            masks_stacked = obs["monomial_masks"]
            poly_mask = obs["poly_masks"]
            selectables_mask = obs["selectables"]

            monomial_embs = jax.vmap(self.monomial_embedder)(ideal_stacked)
            ideal_embeddings = jax.vmap(self.polynomial_embedder)(
                monomial_embs, masks_stacked
            )

            ideal_embeddings = self.ideal_model(ideal_embeddings, mask=poly_mask)

            values = self.pairwise_scorer(ideal_embeddings)

            # Apply selectables mask
            values = values + selectables_mask
            return values.flatten()

        ideal, selectables = obs

        # Pad polynomials to the same length
        ideal_arrays = [jnp.asarray(p) for p in ideal]
        lengths = [p.shape[0] for p in ideal_arrays]
        max_len = max(lengths) if lengths else 0

        padded_ideal = []
        masks = []
        for p in ideal_arrays:
            length = p.shape[0]
            pad_len = max_len - length
            if pad_len > 0:
                p_padded = jnp.pad(p, ((0, pad_len), (0, 0)), constant_values=0)
                mask = jnp.concatenate(
                    [jnp.ones(length, dtype=bool), jnp.zeros(pad_len, dtype=bool)]
                )
            else:
                p_padded = p
                mask = jnp.ones(length, dtype=bool)
            padded_ideal.append(p_padded)
            masks.append(mask)

        ideal_stacked = jnp.stack(padded_ideal)
        masks_stacked = jnp.stack(masks)

        monomial_embs = jax.vmap(self.monomial_embedder)(ideal_stacked)
        ideal_embeddings = jax.vmap(self.polynomial_embedder)(
            monomial_embs, masks_stacked
        )

        poly_mask = jnp.ones(ideal_embeddings.shape[0], dtype=bool)

        ideal_embeddings = self.ideal_model(ideal_embeddings, mask=poly_mask)

        values = self.pairwise_scorer(ideal_embeddings)

        mask = jnp.full(values.shape, -jnp.inf)
        if selectables:
            rows, cols = zip(*selectables)
            rows = jnp.array(rows)
            cols = jnp.array(cols)
            mask = mask.at[rows, cols].set(0.0)

        values = values + mask
        return values.flatten()


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    monomials_dim: int
    monoms_embedding_dim: int = 64
    polys_embedding_dim: int = 128
    poly_embedder_depth: int = 2
    ideal_depth: int = 4
    ideal_backbone: Literal["transformer", "relational"] = "transformer"
    ideal_num_heads: int = 8
    ideal_hidden_dim: int = 256
    value_hidden_dim: int = 128


class GrobnerPolicyValue(Module):
    """
    Neural network model with policy and value heads for RL training.

    The model uses a shared backbone (Extractor) for feature extraction
    and separate heads for policy logits and value estimation.
    Used by AlphaZero, Gumbel MuZero, and Rainbow DQN.

    For DQN the two outputs map directly onto the dueling architecture:
      - policy logits  →  advantage stream  A(s, a)
      - value estimate →  state-value stream V(s)
    Combined via q_values() into Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')].
    """

    extractor: Extractor
    value_head: eqx.nn.MLP

    def __init__(self, extractor: Extractor, value_head: eqx.nn.MLP):
        self.extractor = extractor
        self.value_head = value_head

    @filter_jit
    def embed_polynomial(
        self, poly_tokens: Array, mask: Array | None = None
    ) -> Array:
        """
        Embed a single polynomial into the per-polynomial embedding space.

        Used by callers that maintain an external cache of polynomial
        embeddings: a polynomial that already sits in the basis never changes,
        so its embedding can be computed once and reused across steps. Only
        the contextualization via `ideal_model` has to be recomputed when the
        basis grows.

        Args:
            poly_tokens: Array of shape (num_monomials, monomial_dim).
            mask: Optional bool array of shape (num_monomials,).

        Returns:
            Array of shape (polys_embedding_dim,).
        """
        monomial_embs = self.extractor.monomial_embedder(poly_tokens)
        return self.extractor.polynomial_embedder(monomial_embs, mask)

    @filter_jit
    def policy_value_from_embeddings(
        self,
        poly_embeddings: Array,
        poly_mask: Array,
        selectables_mask: Array,
    ) -> tuple[Array, Array]:
        """
        Apply the ideal-level backbone and the policy/value heads on
        pre-computed polynomial embeddings.

        Args:
            poly_embeddings: (num_polynomials, polys_embedding_dim).
            poly_mask: (num_polynomials,) bool, True for valid polynomials.
            selectables_mask: (num_polynomials, num_polynomials), 0.0 on
                legal pairs and -inf elsewhere.

        Returns:
            (policy_logits, value), with policy_logits flattened to
            (num_polynomials ** 2,).
        """
        ctx = self.extractor.ideal_model(poly_embeddings, mask=poly_mask)
        pair_scores = self.extractor.pairwise_scorer(ctx)
        policy_logits = (pair_scores + selectables_mask).flatten()

        masked_ctx = jnp.where(poly_mask[:, None], ctx, 0.0)
        num_valid = jnp.sum(poly_mask) + 1e-9
        pooled_mean = jnp.sum(masked_ctx, axis=0) / num_valid
        pooled_max = jnp.max(
            jnp.where(poly_mask[:, None], ctx, -jnp.inf), axis=0
        )
        pooled_max = jnp.where(jnp.isneginf(pooled_max), 0.0, pooled_max)
        pooled = jnp.concatenate([pooled_mean, pooled_max], axis=-1)
        value = self.value_head(pooled).squeeze(-1)

        return policy_logits, value

    def __call__(self, obs: Observation | dict | tuple) -> tuple[Array, Array]:
        """
        Forward pass returning policy logits and value estimate.

        Args:
            obs: Environment observation (tuple or dict format).

        Returns:
            Tuple of (policy_logits, value).
        """
        if isinstance(obs, dict):
            ideal_stacked = obs["ideals"]
            masks_stacked = obs["monomial_masks"]
            poly_mask = obs["poly_masks"]
            selectables_mask = obs["selectables"]
        else:
            ideal, selectables = obs

            polys_np = [np.asarray(p) for p in ideal]
            n_polys_raw = len(polys_np)
            max_monoms_raw = max((p.shape[0] for p in polys_np), default=1)
            n_vars = polys_np[0].shape[1] if polys_np else 0

            n_polys = _next_pow2(n_polys_raw)
            max_monoms = _next_pow2(max_monoms_raw)

            ideal_np = np.zeros((n_polys, max_monoms, n_vars), dtype=np.float32)
            mono_mask_np = np.zeros((n_polys, max_monoms), dtype=bool)
            poly_mask_np = np.zeros(n_polys, dtype=bool)
            for i, p in enumerate(polys_np):
                length = p.shape[0]
                ideal_np[i, :length] = p
                mono_mask_np[i, :length] = True
                poly_mask_np[i] = True

            selectables_np = np.full((n_polys, n_polys), -np.inf, dtype=np.float32)
            if selectables:
                rows, cols = zip(*selectables)
                selectables_np[list(rows), list(cols)] = 0.0

            ideal_stacked = jnp.asarray(ideal_np)
            masks_stacked = jnp.asarray(mono_mask_np)
            poly_mask = jnp.asarray(poly_mask_np)
            selectables_mask = jnp.asarray(selectables_np)

        poly_embeddings = jax.vmap(self.embed_polynomial)(
            ideal_stacked, masks_stacked
        )
        return self.policy_value_from_embeddings(
            poly_embeddings, poly_mask, selectables_mask
        )

    def q_values(self, obs: "Observation | dict | tuple") -> Array:
        """
        Dueling Q-values for a single observation.

        Combines the advantage stream (policy logits) and the state-value
        estimate into per-action Q-values:
            Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

        Args:
            obs: Environment observation (tuple or dict format).

        Returns:
            Array of Q-values with the same shape as the policy logits.
        """
        advantages, value = self(obs)
        valid_mask = jnp.isfinite(advantages)
        valid_advantages = jnp.where(valid_mask, advantages, 0.0)
        mean_advantage = valid_advantages.sum() / (valid_mask.sum() + 1e-9)
        return value + (advantages - mean_advantage)

    @classmethod
    def from_scratch(cls, config: ModelConfig, key: Array) -> "GrobnerPolicyValue":
        """Initialize model from scratch."""
        keys = jax.random.split(key, 5)
        k_monomial, k_polynomial, k_ideal, k_scorer, k_value = keys

        monomial_embedder = MonomialEmbedder(
            config.monomials_dim, config.monoms_embedding_dim, k_monomial
        )
        polynomial_embedder = PolynomialEmbedder(
            input_dim=config.monoms_embedding_dim,
            hidden_dim=config.polys_embedding_dim,
            hidden_layers=config.poly_embedder_depth,
            output_dim=config.polys_embedding_dim,
            key=k_polynomial,
        )
        ideal_model: IdealModel | RelationalIdealModel
        if config.ideal_backbone == "transformer":
            ideal_model = IdealModel(
                config.polys_embedding_dim,
                config.ideal_num_heads,
                config.ideal_depth,
                k_ideal,
            )
        else:
            ideal_model = RelationalIdealModel(
                config.polys_embedding_dim,
                config.ideal_hidden_dim,
                config.ideal_depth,
                k_ideal,
            )
        pairwise_scorer = PairwiseScorer(
            config.polys_embedding_dim, config.polys_embedding_dim, k_scorer
        )
        extractor = Extractor(
            monomial_embedder, polynomial_embedder, ideal_model, pairwise_scorer
        )

        value_head = eqx.nn.MLP(
            in_size=config.polys_embedding_dim * 2,
            out_size=1,
            width_size=config.value_hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=k_value,
        )

        return cls(extractor=extractor, value_head=value_head)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: ModelConfig,
        optimizer: optax.GradientTransformation,
        key: Array,
    ) -> "GrobnerPolicyValue":
        """
        Initialize from a pretrained GrobnerPolicyValue checkpoint.

        Loads the complete GrobnerPolicyValue model including both
        extractor and value head weights.

        Args:
            checkpoint_path: Path to pretrained checkpoint.
            config: Model configuration.
            optimizer: Optimizer for creating template opt_state.
            key: JAX random key (used for template creation only).

        Returns:
            GrobnerPolicyValue with pretrained weights.
        """
        from grobnerRl.training.utils import load_checkpoint

        template_model = cls.from_scratch(config, key)
        template_opt_state = optimizer.init(eqx.filter(template_model, eqx.is_array))

        template = {
            "model": template_model,
            "opt_state": template_opt_state,
            "iteration": 0,
            "metrics": 0.0,
        }

        payload = load_checkpoint(checkpoint_path, template)

        return payload["model"]
