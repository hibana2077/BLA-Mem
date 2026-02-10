import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

class ParityDataset(IterableDataset):
    def __init__(self, seq_len, batch_size=32, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        while True:
            # Input: (B, T, 1) random 0 or 1
            x = torch.randint(0, 2, (self.batch_size, self.seq_len, 1), device=self.device).float()
            # Label: sum mod 2 along sequence
            y = x.sum(dim=1).long() % 2
            y = y.squeeze(-1) # (B,)
            yield x, y

class AddingDataset(IterableDataset):
    def __init__(self, seq_len, batch_size=32, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        while True:
            # Input: (B, T, 2). Channel 0: values U[-1, 1]. Channel 1: mask.
            values = torch.rand(self.batch_size, self.seq_len, 1, device=self.device) * 2 - 1
            
            # Create mask with exactly two 1s
            indices = torch.argsort(torch.rand(self.batch_size, self.seq_len, device=self.device), dim=1)[:, :2]
            mask = torch.zeros(self.batch_size, self.seq_len, 1, device=self.device)
            mask.scatter_(1, indices.unsqueeze(-1), 1.0)
            
            x = torch.cat([values, mask], dim=-1) # (B, T, 2)
            
            # Label: sum of values where mask is 1
            y = (values * mask).sum(dim=1).squeeze(-1) # (B,)
            yield x, y
