import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils import enable_mc_dropout


@torch.no_grad()
def compute_predictive_entropy(model, forget_train_ds, device, num_mc_samples=30, batch_size=128):
    loader = DataLoader(forget_train_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    enable_mc_dropout(model)

    uncertainties = []
    for images, _ in tqdm(loader, desc="Computing uncertainty"):
        images = images.to(device)
        logits_mc = torch.stack([model(images) for _ in range(num_mc_samples)], dim=0)
        probs_mc = F.softmax(logits_mc, dim=-1)
        mean_probs = probs_mc.mean(dim=0)

        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        uncertainties.extend(entropy.cpu().numpy())

    return np.array(uncertainties)


def get_forget_loader(forget_train_ds, uncertainties, mode="curriculum_low_first", batch_size=32):
    idx = np.arange(len(forget_train_ds))

    if mode == "random":
        np.random.shuffle(idx)
    elif mode == "curriculum_low_first":
        idx = idx[np.argsort(uncertainties)]
    elif mode == "curriculum_high_first":
        idx = idx[np.argsort(-uncertainties)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return DataLoader(Subset(forget_train_ds, idx), batch_size=batch_size, shuffle=False)