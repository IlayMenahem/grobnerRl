import torch
from torch import nn
import numpy as np


class Extractor(nn.Module):
    monomial_embedder: nn.Module
    polynomial_transformer: nn.Module
    polynomial_embedder: nn.Module
    ideal_transformer: nn.Module

    def __init__(self, num_vars: int, monoms_embedding_dim: int, polys_embedding_dim: int,
        polys_depth: int, polys_num_heads: int, ideal_depth: int, ideal_num_heads: int, dropout: float = 0.0):

        super(Extractor, self).__init__()

        self.monomial_embedder = nn.Linear(num_vars, monoms_embedding_dim)
        self.polynomial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(monoms_embedding_dim, polys_num_heads, dropout=dropout, batch_first=True),
            num_layers=polys_depth
        )
        self.polynomial_embedder = nn.Linear(monoms_embedding_dim, polys_embedding_dim)
        self.ideal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(polys_embedding_dim, ideal_num_heads,  dropout=dropout, batch_first=True),
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
        ideal = [torch.Tensor(polynomial) for polynomial in ideal]

        # Embed monomials
        monomial_embeddings = [self.monomial_embedder(polynomial) for polynomial in ideal]

        # Embed polynomials
        # Process each polynomial's monomials as a batch of size 1
        polynomial_encodings = [self.polynomial_transformer(monomial_embedding.unsqueeze(0)).squeeze(0) for monomial_embedding in monomial_embeddings]
        polynomial_encodings = [torch.mean(polynomial_encoding, dim=0) for polynomial_encoding in polynomial_encodings]
        polynomial_encodings = torch.stack(polynomial_encodings)
        polynomial_encodings = self.polynomial_embedder(polynomial_encodings)

        # Embed ideals
        # Process the ideal's polynomials as a batch of size 1
        ideal_embeddings = self.ideal_transformer(polynomial_encodings.unsqueeze(0)).squeeze(0)
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
            values = [self.forward(o) for o in obs]
            return torch.stack(values)

        ideal, _ = obs

        vals = self.extractor(ideal)
        value = torch.mean(vals)

        return value


class TwinExtractor(nn.Module):
    '''
    gets a pair polynomials and returns the value of the pair
    '''

    def __init__(self, num_vars: int, num_monomials: int, polys_embedding_dim: int):
        super(TwinExtractor, self).__init__()
        self.linear1 = nn.Linear(num_vars * num_monomials, polys_embedding_dim)
        self.linear2 = nn.Linear(polys_embedding_dim, polys_embedding_dim)
        self.evaluation1 = nn.Linear(2*polys_embedding_dim, polys_embedding_dim)
        self.evaluation2 = nn.Linear(polys_embedding_dim, 1)


    def _embed_polynomial(self, poly: torch.Tensor) -> torch.Tensor:
        '''
        embeds a polynomial represented as a tensor

        Args:
        - poly (torch.Tensor): a tensor representing a polynomial

        Returns:
        - torch.Tensor: the embedded polynomial
        '''
        poly = poly.flatten()

        if poly.shape[0] != self.linear1.in_features:
            poly = torch.cat((poly, torch.zeros(self.linear1.in_features - poly.shape[0], dtype=poly.dtype)))

        x = self.linear1(poly)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        '''
        takes a pair of polynomials and returns the value of reducing the pair

        Args:
        - pair (tuple[torch.Tensor, torch.Tensor]): a pair of polynomials, each represented as a tensor

        Returns:
        - torch.Tensor: the value of the pair
        '''
        poly1, poly2 = pair

        poly1 = self._embed_polynomial(poly1)
        poly2 = self._embed_polynomial(poly2)

        x = torch.cat((poly1, poly2), dim=0)
        x = self.evaluation1(x)
        x = torch.relu(x)
        x = self.evaluation2(x)

        return x.squeeze(0)


class TwinPolicy(nn.Module):
    def __init__(self, extractor: TwinExtractor):
        super(TwinPolicy, self).__init__()
        self.extractor = extractor

    def forward(self, obs: tuple|list[tuple]) -> torch.Tensor| list[torch.Tensor]:
        if isinstance(obs, list):
            res = [self.forward(o) for o in obs]

            return res

        ideal, selectables = obs
        vals = torch.full((len(ideal), len(ideal)), float('-inf'))

        for i, j in selectables:
            vals[i, j] = self.extractor((torch.Tensor(ideal[i]), torch.Tensor(ideal[j])))

        vals = vals.flatten()
        probs = torch.softmax(vals, dim=0)

        return probs


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
