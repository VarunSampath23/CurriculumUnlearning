import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_mc_dropout(model: nn.Module):
    """Enable dropout layers for MC Dropout inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")