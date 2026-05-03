import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


def entropy(p, dim=-1):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim)


def collect_prob(data_loader, model):
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(next(model.parameters()).device)
            output = model(x)
            probs.append(F.softmax(output, dim=-1))
    return torch.cat(probs)


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    n_test = len(test_prob)
    idx = torch.randperm(len(retain_prob))[:n_test]
    retain_balanced = retain_prob[idx]

    # Simple entropy threshold attack
    retain_entropy = entropy(retain_balanced).cpu().numpy()
    test_entropy = entropy(test_prob).cpu().numpy()
    threshold = retain_entropy.mean()

    forget_entropy = entropy(forget_prob).cpu().numpy().reshape(-1)
    predictions = (forget_entropy < threshold).astype(int)
    return predictions.mean()  # attack accuracy