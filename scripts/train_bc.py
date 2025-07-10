import os
import torch
from torch.utils.data import DataLoader
from grobnerRl.data import BCDataset, bc_collate
from grobnerRl.models import GrobnerPolicy, Extractor
from implementations.bc import train_model, bc_loss, accuracy_fn


if __name__ == "__main__":
    num_vars = 3
    num_monomials = 2
    max_degree = 4
    num_polynomials = 4
    path = os.path.join(os.getcwd(), 'data', 'optimal_reductions.json')

    dataset = BCDataset(path)
    dataloader_train = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=bc_collate)
    dataloader_val = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=bc_collate)
    policy = GrobnerPolicy(Extractor(3, 32, 128, 2, 4))
    # policy = TwinPolicy(TwinExtractor(num_vars, num_monomials, 256))

    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)
    schedualer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)

    policy = train_model(policy, dataloader_train, dataloader_val, 500, optimizer,
        bc_loss, accuracy_fn, scheduler=schedualer)
