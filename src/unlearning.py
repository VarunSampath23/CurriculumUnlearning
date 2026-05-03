import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def gradient_ascent_unlearning(model, forget_loader, device, lr=5e-6, epochs=5):
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(forget_loader, desc=f"GA Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = -criterion(model(images), labels)   # negative loss = ascent
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[GA] Epoch {epoch+1}/{epochs} | Avg Ascent Loss: {running_loss/len(forget_loader):.4f}")

    return model