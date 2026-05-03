import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return train_transform, test_transform


def load_cifar10_data(data_root="./data", batch_size=64, num_workers=4):
    train_transform, test_transform = get_cifar10_transforms()

    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return train_ds, test_ds


def create_forget_retain_datasets(train_ds, test_ds, forget_class: int = 1):
    train_targets = train_ds.targets
    test_targets = test_ds.targets

    forget_train_idx = [i for i, y in enumerate(train_targets) if y == forget_class]
    retain_train_idx = [i for i, y in enumerate(train_targets) if y != forget_class]

    forget_test_idx = [i for i, y in enumerate(test_targets) if y == forget_class]
    retain_test_idx = [i for i, y in enumerate(test_targets) if y != forget_class]

    forget_train_ds = Subset(train_ds, forget_train_idx)
    retain_train_ds = Subset(train_ds, retain_train_idx)
    forget_test_ds = Subset(test_ds, forget_test_idx)
    retain_test_ds = Subset(test_ds, retain_test_idx)

    print(f"Forget Train: {len(forget_train_ds)} | Retain Train: {len(retain_train_ds)}")
    print(f"Forget Test : {len(forget_test_ds)} | Retain Test : {len(retain_test_ds)}")

    return forget_train_ds, retain_train_ds, forget_test_ds, retain_test_ds