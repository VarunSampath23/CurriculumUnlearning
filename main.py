import argparse
import os
import torch
from torch.utils.data import DataLoader
from src.utils import set_seed, get_device
from src.dataset import load_cifar10_data, create_forget_retain_datasets
from src.model import get_resnet18
from src.curriculum import compute_predictive_entropy, get_forget_loader
from src.unlearning import gradient_ascent_unlearning
from src.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Uncertainty Curriculum Machine Unlearning")
    parser.add_argument("--forget_class", type=int, default=1, help="Class to forget")
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--unlearn_epochs", type=int, default=5)
    parser.add_argument("--unlearn_lr", type=float, default=5e-6)
    parser.add_argument("--mode", type=str, default="curriculum_low_first",
                        choices=["random", "curriculum_low_first", "curriculum_high_first"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = f"{args.save_dir}/checkpoints/pretrained_resnet18_cifar10.pt"

    # Data
    train_ds, test_ds = load_cifar10_data()
    forget_train_ds, retain_train_ds, forget_test_ds, retain_test_ds = create_forget_retain_datasets(
        train_ds, test_ds, args.forget_class
    )

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    retain_train_dl = DataLoader(retain_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    retain_test_dl = DataLoader(retain_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    forget_test_dl = DataLoader(forget_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_resnet18().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded pretrained model")
    else:
        print("No checkpoint found – please train first")

    # Before unlearning
    print("\n=== BEFORE UNLEARNING ===")
    run_full_evaluation(model, retain_test_dl, forget_test_dl, retain_train_dl,
                        DataLoader(forget_train_ds, batch_size=args.batch_size, shuffle=True),
                        test_dl, device, prefix="[Before]")

    # Uncertainty + Curriculum
    uncertainties = compute_predictive_entropy(model, forget_train_ds, device,
                                               num_mc_samples=args.mc_samples)
    forget_loader = get_forget_loader(forget_train_ds, uncertainties, args.mode, args.batch_size)

    # Unlearning
    model = gradient_ascent_unlearning(model, forget_loader, device,
                                       lr=args.unlearn_lr, epochs=args.unlearn_epochs)

    # After unlearning
    print("\n=== AFTER UNLEARNING ===")
    run_full_evaluation(model, retain_test_dl, forget_test_dl, retain_train_dl,
                        DataLoader(forget_train_ds, batch_size=args.batch_size, shuffle=True),
                        test_dl, device, prefix=f"[{args.mode}]")

    # Save final model
    torch.save(model.state_dict(), f"{args.save_dir}/checkpoints/final_{args.mode}.pt")


if __name__ == "__main__":
    main()