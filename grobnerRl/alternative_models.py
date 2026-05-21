import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


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
