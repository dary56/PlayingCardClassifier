import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import SimpleCardClassifier
from config import CHECKPOINT_PATH, LOSSES_PATH, CLASSES_PATH


def train(num_epochs=5, batch_size=32, lr=0.001):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, valid_loader, _, class_names = get_dataloaders(batch_size=batch_size)

    model = SimpleCardClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation loop'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved new best model (valid_loss={valid_loss:.4f})")

        with open(LOSSES_PATH, 'w') as f:
            json.dump({'train': train_losses, 'valid': valid_losses}, f)

    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f)

    print(f"\nDone. Best model: {CHECKPOINT_PATH}, losses: {LOSSES_PATH}")


if __name__ == "__main__":
    train()