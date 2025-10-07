import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor|None = None) -> torch.Tensor:
        """
        x: Tensor of shape (n, d) or (batch_size, n, d)
        mask: Optional tensor of shape (batch_size, n) for padded sequences
        returns: Tensor of shape (d',) or (batch_size, d',)
        """
        h = self.phi(x)

        if mask is not None:
            # Apply mask to ignore padded elements
            h = h * mask.unsqueeze(-1)
            h_sum = torch.sum(h, dim=-2)
        else:
            h_sum = torch.sum(h, dim=-2 if x.dim() == 3 else 0)

        res = self.rho(h_sum)

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
            nn.TransformerEncoderLayer(polys_embedding_dim, ideal_num_heads, dim_feedforward=2*polys_embedding_dim, dropout=0.1, batch_first=True),
            num_layers=ideal_depth
        )

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor:
        if isinstance(obs, list):
            return pad_sequence([self.forward(o) for o in obs], batch_first=True)

        ideal: list[np.ndarray] = obs[0]
        selectables: list[tuple[int,int]] = obs[1]

        # make batch a padded tensor
        _ideal_tensors = [torch.as_tensor(poly, dtype=torch.float32) for poly in ideal]
        _ideal_padded = pad_sequence(_ideal_tensors, batch_first=True)

        # Create mask to ignore padded elements
        lengths = torch.as_tensor([len(poly) for poly in ideal])
        max_len = _ideal_padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

        polynomial_encodings = self.polynomial_embedder(_ideal_padded, mask)

        ideal_embeddings = self.ideal_transformer(polynomial_encodings.unsqueeze(0)).squeeze(0)
        values = torch.matmul(ideal_embeddings, ideal_embeddings.T)

        vals = apply_mask(values, selectables)

        return vals


class GrobnerPolicy(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerPolicy, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor:
        vals = self.extractor(obs)

        if isinstance(obs, list):
            return torch.softmax(vals, dim=-1)

        probs = torch.softmax(vals, dim=0)

        return probs


class GrobnerValue(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerValue, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor:
        vals = self.extractor(obs)

        return vals


class GrobnerCritic(nn.Module):
    extractor: nn.Module

    def __init__(self, extractor: nn.Module):
        super(GrobnerCritic, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor:
        if isinstance(obs, list):
            return torch.stack([self.forward(o) for o in obs])

        vals = self.extractor(obs)
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
