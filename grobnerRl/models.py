import torch
from torch import nn
import numpy as np


class DeepSetsEncoder(nn.Module):
    """
    A permutation-invariant encoder for sets implemented in PyTorch.

    Given input X of shape (n, d), this module outputs a vector of shape (d',).
    Internally it applies phi to each row, sums across the set dimension,
    then applies rho to the aggregated representation.
    """
    phi: nn.Module
    rho: nn.Module

    def __init__(self, input_dim: int, phi_hidden: int, rho_hidden: int, output_dim: int):
        super(DeepSetsEncoder, self).__init__()
        # phi: elementwise feature extractor
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden)
        )
        # rho: post-aggregation processor
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Nested tensor with jagged layout or regular tensor
        returns: Tensor of shape (d',) or (batch_size, d',)
        """
        components = x.unbind()
        processed = []

        for comp in components:
            h = self.phi(comp)
            h_sum = h.sum(dim=0)
            processed.append(h_sum)

        stacked = torch.stack(processed)
        res = self.rho(stacked)

        return res


def apply_mask(vals: torch.Tensor, selectables: list[tuple[int,int]]) -> torch.Tensor:
    mask = torch.full_like(vals, float('-inf'))

    if selectables:
        rows, cols = zip(*selectables)
        mask[rows, cols] = 0.0

    vals = vals + mask
    vals = vals.flatten()

    return vals


class Extractor(nn.Module):
    polynomial_embedder: DeepSetsEncoder
    ideal_transformer: nn.Module

    def __init__(self, num_vars: int, monoms_embedding_dim: int, polys_embedding_dim: int, ideal_depth: int, ideal_num_heads: int):

        super(Extractor, self).__init__()

        self.polynomial_embedder = DeepSetsEncoder(num_vars, monoms_embedding_dim, polys_embedding_dim, polys_embedding_dim)
        self.ideal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(polys_embedding_dim, ideal_num_heads, dim_feedforward=4*polys_embedding_dim, dropout=0.1, batch_first=True),
            num_layers=ideal_depth
        )

    def forward(self, obs: tuple|list) -> torch.Tensor:
        """
        Forward pass that handles batched nested tensors.

        Args:
            obs: tuple of (nested_batch, selectables_batch) where:
                - nested_batch: list of nested tensors (one per batch item)
                - selectables_batch: list of selectables for each batch item

        Returns:
            List of tensors (for batched input) or single tensor
        """
        def process_single(nested_ideal, selectables):
            polynomial_encodings = self.polynomial_embedder(nested_ideal)
            ideal_embeddings = self.ideal_transformer(polynomial_encodings.unsqueeze(0)).squeeze(0)
            values = torch.matmul(ideal_embeddings, ideal_embeddings.T)
            vals = apply_mask(values, selectables)

            return vals

        nested_batch, selectables_batch = obs

        if isinstance(nested_batch, list):
            batch_outputs = []

            for nested_ideal, selectables in zip(nested_batch, selectables_batch):
                vals = process_single(nested_ideal, selectables)
                batch_outputs.append(vals)

            return batch_outputs
        else:
            vals = process_single(nested_batch, selectables_batch)

            return vals


class GrobnerPolicy(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerPolicy, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list) -> torch.Tensor:
        # Return logits for training, not probabilities
        vals = self.extractor(obs)
        return vals


class GrobnerValue(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerValue, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list) -> torch.Tensor:
        vals = self.extractor(obs)

        return vals


class GrobnerCritic(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerCritic, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list) -> torch.Tensor:
        vals = self.extractor(obs)

        if isinstance(vals, list):
            # Batched output - take max of each
            return torch.stack([torch.max(v) for v in vals])

        max_val = torch.max(vals)

        return max_val


if __name__ == "__main__":
    num_vars = 3
    num_monomials = 2
    monoms_embedding_dim = 32
    polys_embedding_dim = 64
    ideal_depth = 2
    ideal_num_heads = 4

    extractor_policy = Extractor(num_vars, monoms_embedding_dim, polys_embedding_dim,
                          ideal_depth, ideal_num_heads)
    extractor_critic = Extractor(num_vars, monoms_embedding_dim, polys_embedding_dim,
                            ideal_depth, ideal_num_heads)

    policy = GrobnerPolicy(extractor_policy)
    critic = GrobnerCritic(extractor_critic)

    # Using numpy arrays as per the forward method's type hint
    ideal = [[np.random.randn(num_vars+1) for _ in range(num_monomials)] for i in range(1,4)]
    selectables = [(0, 1), (0, 2)]

    obs = (ideal, selectables)

    print("Policy output:")
    print(policy(obs))
    print("\nCritic output:")
    print(critic(obs))
