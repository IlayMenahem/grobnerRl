import torch
from torch import nn


class Extractor(nn.Module):
    MonomialEmbedder: nn.Module
    PolynomialTransformer: nn.Module
    Polynomial_embedder: nn.Module
    IdealTransformer: nn.Module

    def __init__(self, num_vars: int, monoms_embedding_dim: int, polys_embedding_dim: int,
        polys_depth: int, polys_num_heads: int, ideal_depth: int, ideal_num_heads: int, dropout: float = 0.0):

        super(Extractor, self).__init__()

        self.MononomialEmbedder = nn.Linear(num_vars, monoms_embedding_dim)
        self.PolynomialTransformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(monoms_embedding_dim, polys_num_heads, dim_feedforward = monoms_embedding_dim, dropout=dropout),
            num_layers=polys_depth
        )
        self.Polynomial_embedder = nn.Linear(monoms_embedding_dim, polys_embedding_dim)
        self.IdealTransformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(polys_embedding_dim, ideal_num_heads, dim_feedforward = polys_embedding_dim,  dropout=dropout),
            num_layers=ideal_depth
        )

    def forward(self, ideal: list[list[torch.Tensor]]) -> torch.Tensor:
        '''
        Args:
        ideal: list of lists of tensors, where each tensor represents a monomial

        Returns:
        torch.Tensor - values of the selectable pairs, the non selectable pairs are
        set to -inf
        '''
        ideal = [[torch.Tensor(monomial) for monomial in polynomial] for polynomial in ideal]

        # Embed monomials
        monomial_embeddings = [[self.MononomialEmbedder(monomial) for monomial in polynomial] for polynomial in ideal]
        monomial_embeddings = [torch.stack(monomials) for monomials in monomial_embeddings]

        # Embed polynomials
        polynomial_encodings = [self.PolynomialTransformer(monomial_embedding.unsqueeze(0)).squeeze(0) for monomial_embedding in monomial_embeddings]
        polynomial_encodings = [torch.mean(polynomial_encoding, dim=0) for polynomial_encoding in polynomial_encodings]
        polynomial_encodings = torch.stack(polynomial_encodings)
        polynomial_encodings = self.Polynomial_embedder(polynomial_encodings)

        # Embed ideals
        ideal_embeddings = self.IdealTransformer(polynomial_encodings.unsqueeze(1)).squeeze(1)
        values = torch.matmul(ideal_embeddings, ideal_embeddings.T)

        return values


class GrobnerPolicy(nn.Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        super(GrobnerPolicy, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor| list[torch.Tensor]:
        if isinstance(obs, list):
            return [self.forward(o) for o in obs]

        ideal: list[list[torch.Tensor]] = obs[0]
        selectables: list[tuple[int,int]] = obs[1]
        vals = self.extractor(ideal)

        mask = torch.full_like(vals, float('-inf'))
        for i, j in selectables:
            mask[i, j] = 0.0

        vals = vals + mask
        vals = vals.flatten()

        probs = torch.softmax(vals, dim=0)

        return probs


class GrobnerCritic(nn.Module):
    extractor: Extractor

    def __init__(self, extractor: Extractor):
        super(GrobnerCritic, self).__init__()

        self.extractor = extractor

    def forward(self, obs: tuple) -> torch.Tensor:
        if isinstance(obs, list):
            return [self.forward(o) for o in obs]

        ideal, _ = obs

        vals = self.extractor(ideal)
        value = torch.mean(vals)

        return value


if __name__ == "__main__":
    num_vars = 3
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

    ideal = [[torch.randn(num_vars) for _ in range(i)] for i in range(1,4)]
    selectables = [(0, 1), (1, 2), (2, 0)]

    obs = (ideal, selectables)

    print(policy(obs))
    print(critic(obs))
