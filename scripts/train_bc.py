import os
import torch
from torch.utils.data import DataLoader
from grobnerRl.data import BCDataset, bc_collate
from grobnerRl.models import GrobnerPolicy, Extractor
from grobnerRl.rl.bc import train_bc


if __name__ == "__main__":
    num_vars = 3
    max_degree = 4
    num_polynomials = 4
    path = os.path.join(os.getcwd(), 'data', 'optimal_reductions.json')

    dataset = BCDataset(path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=bc_collate)
    policy = GrobnerPolicy(Extractor(3, 32, 128, 2, 4, 2, 4))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    policy = train_bc(policy, dataloader, 100, optimizer)
