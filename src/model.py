import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .utils import set_seed


def get_resnet18(num_classes=10, dropout_p=0.5, seed=42):
    set_seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model