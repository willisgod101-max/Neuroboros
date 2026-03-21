import torch
from torch.utils.data import Dataset
from typing import Tuple

class CharDataset(Dataset):
    def __init__(self, data: str, block_size: int):
        chars = sorted(list(set(data)))
        data_size = len(data)
        vocab_size = len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # pick a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y