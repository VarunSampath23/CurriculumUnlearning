#!/usr/bin/env python3
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, get_device, evaluate
from src.dataset import load_cifar10_data, create_forget_retain_datasets
from src.model import get_resnet18


def train_model(
    train_dl: DataLoader,
    test_dl: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 5e-4,
    save_path: str = None,
    seed: int = 42,
):
    set_seed(seed)
    model = get_resnet18().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        acc = evaluate(model, test_dl, device)

        print(
            f"Epoch {epoch+1:2d} | "
            f"Loss {running_loss/len(train_dl):.4f} | "
            f"Test Acc {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    # Save best model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path} (Acc: {best_acc:.2f}%)")

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10 (pre-unlearning)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="outputs/checkpoints/pretrained_resnet18_cifar10.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Data
    train_ds, test_ds = load_cifar10_data()
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Train
    train_model(
        train_dl=train_dl,
        test_dl=test_dl,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()