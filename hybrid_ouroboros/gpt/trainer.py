import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
from dataclasses import dataclass
from typing import Tuple, Optional

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

@dataclass
class TrainerConfig:
    # optimization parameters
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    weight_decay: float = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay: bool = False
    warmup_tokens: int = 375_000_000  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens: int = 260_000_000_000   # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path: Optional[str] = None
    num_workers: int = 0  # for DataLoader
    # logging
    log_interval: int = 100
    save_interval: int = 1  # save every epoch

def train(model: nn.Module, train_dataset: CharDataset, config: TrainerConfig, device: str) -> nn.Module:
    model = model.to(device)
    model.train()

    # create a DataLoader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # create an PyTorch optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=config.betas,
        device_type=device,
    )

    # training loop
    for epoch in range(config.max_epochs):
        epoch_start_time = time.time()
        losses = []
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            # forward the model
            logits, loss = model(x, y)
            losses.append(loss.item())
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()

            # logging
            if it % config.log_interval == 0:
                print(f"epoch [{epoch+1}/{config.max_epochs}] iter {it}: loss {loss.item():.4f}")

        epoch_end_time = time.time()
        print(f"epoch [{epoch+1}/{config.max_epochs}] completed in {epoch_end_time - epoch_start_time:.2f}s, loss: {sum(losses)/len(losses):.4f}")

        # checkpointing
        if (epoch + 1) % config.save_interval == 0 and config.ckpt_path is not None:
            checkpoint_path = os.path.join(config.ckpt_path, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    return model

if __name__ == '__main__':
    # This is just a placeholder for testing the trainer
    print("Trainer module for GPT")