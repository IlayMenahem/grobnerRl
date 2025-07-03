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
            nn.Linear(rho_hidden, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (n, d)
        returns: Tensor of shape (d',)
        """
        h = self.phi(x)
        h_sum = torch.sum(h, dim=0)
        res = self.rho(h_sum)

        return res


class Extractor(nn.Module):
    polynomial_embedder: DeepSetsEncoder
    ideal_transformer: nn.Module

    def __init__(self, num_vars: int, monoms_embedding_dim: int, polys_embedding_dim: int, ideal_depth: int, ideal_num_heads: int):

        super(Extractor, self).__init__()

        self.polynomial_embedder = DeepSetsEncoder(num_vars, monoms_embedding_dim, polys_embedding_dim, polys_embedding_dim)
        self.ideal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(polys_embedding_dim, ideal_num_heads,  dropout=0.0, batch_first=True),
            num_layers=ideal_depth
        )

    def forward(self, ideal: list[list[np.ndarray]]) -> torch.Tensor:
        '''
        Args:
        ideal: list of lists of tensors, where each tensor represents a monomial

        Returns:
        torch.Tensor - values of the selectable pairs, the non selectable pairs are
        set to -inf
        '''
        _ideal = [torch.Tensor(polynomial) for polynomial in ideal]

        # Embed polynomials
        polynomial_encodings = [self.polynomial_embedder(poly) for poly in _ideal]
        polynomial_encodings = torch.stack(polynomial_encodings)

        # Embed ideals
        # Process the ideal's polynomials as a batch of size 1
        ideal_embeddings = self.ideal_transformer(polynomial_encodings.unsqueeze(0)).squeeze(0)
        values = torch.matmul(ideal_embeddings, ideal_embeddings.T)

        return values


def apply_mask(vals, selectables):
    mask = torch.full_like(vals, float('-inf'))
    for i, j in selectables:
        mask[i, j] = 0.0

    vals = vals + mask
    vals = vals.flatten()

    return vals


class GrobnerPolicy(nn.Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        super(GrobnerPolicy, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor| list[torch.Tensor]:
        if isinstance(obs, list):
            return pad_sequence([self.forward(o) for o in obs], batch_first=True)

        ideal: list[list[torch.Tensor]] = obs[0]
        selectables: list[tuple[int,int]] = obs[1]

        vals = self.extractor(ideal)
        vals = apply_mask(vals, selectables)

        probs = torch.softmax(vals, dim=0)

        return probs


class GrobnerEvaluator(nn.Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        super(GrobnerEvaluator, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor|list[torch.Tensor]:
        if isinstance(obs, list):
            return pad_sequence([self.forward(o) for o in obs], batch_first=True)

        ideal: list[list[torch.Tensor]] = obs[0]
        selectables: list[tuple[int,int]] = obs[1]

        vals = self.extractor(ideal)
        vals = apply_mask(vals, selectables)

        return vals


class GrobnerCritic(nn.Module):
    polynomial_embedder: DeepSetsEncoder
    ideal_encoder: DeepSetsEncoder
    evaluator: nn.Module

    def __init__(self, num_vars: int, monoms_embedding_dim: int, polys_embedding_dim: int,
       ideal_encodeing_dim: int):
        super(GrobnerCritic, self).__init__()

        self.polynomial_embedder = DeepSetsEncoder(num_vars, monoms_embedding_dim, polys_embedding_dim, polys_embedding_dim)
        self.ideal_encoder = DeepSetsEncoder(polys_embedding_dim, polys_embedding_dim, ideal_encodeing_dim, ideal_encodeing_dim)

        self.evaluator = nn.Sequential(
            nn.Linear(ideal_encodeing_dim, ideal_encodeing_dim),
            nn.ReLU(),
            nn.Linear(ideal_encodeing_dim, 1)
        )

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor|list[torch.Tensor]:
        if isinstance(obs, list):
            values = torch.stack([self.forward(o) for o in obs])
            return values

        ideal, _ = obs
        ideal = [torch.Tensor(polynomial) for polynomial in ideal]

        # Embed polynomials
        polynomial_encodings = [self.polynomial_embedder(poly) for poly in ideal]
        polynomial_encodings = torch.stack(polynomial_encodings)

        # Embed ideals
        ideal_embeddings = self.ideal_encoder(polynomial_encodings)

        # Evaluate the ideal
        values = self.evaluator(ideal_embeddings)

        return values


if __name__ == "__main__":
    num_vars = 3
    num_monomials = 2
    monoms_embedding_dim = 32
    polys_embedding_dim = 64
    polys_depth = 2
    polys_num_heads = 4
    ideal_depth = 2
    ideal_num_heads = 4

    extractor_policy = Extractor(num_vars, monoms_embedding_dim, polys_embedding_dim,
                          polys_depth, polys_num_heads, ideal_depth, ideal_num_heads)
    extractor_critic = Extractor(num_vars, monoms_embedding_dim, polys_embedding_dim,
                            polys_depth, polys_num_heads, ideal_depth, ideal_num_heads)

    policy = GrobnerPolicy(extractor_policy)
    critic = GrobnerCritic(extractor_critic)

    # Using numpy arrays as per the forward method's type hint
    ideal = [[np.random.randn(num_vars) for _ in range(num_monomials)] for i in range(1,4)]
    selectables = [(0, 1), (1, 2), (2, 0)]

    obs = (ideal, selectables)

    print("Policy output:")
    print(policy(obs))
    print("\nCritic output:")
    print(critic(obs))

    twin_extractor = TwinExtractor(num_vars, num_monomials, polys_embedding_dim)
    twin_policy = TwinPolicy(twin_extractor)

    print("\nTwin Policy output:")
    print(twin_policy(obs))
