import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import SimpleCardClassifier
from config import CHECKPOINT_PATH, LOSSES_PATH, CLASSES_PATH, BEST_PARAMS_PATH


def create_optimizer(model, optimizer_name, lr, weight_decay=0.0):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train(num_epochs=5, batch_size=32, lr=0.001,
          optimizer_name='adam', weight_decay=0.0,
          save_checkpoint=True, save_losses=True, verbose=True):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    train_loader, valid_loader, _, class_names = get_dataloaders(batch_size=batch_size)

    model = SimpleCardClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_name, lr, weight_decay)

    train_losses, valid_losses = [], []
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_iterator = tqdm(train_loader, desc='Training loop') if verbose else train_loader
        for images, labels in train_iterator:
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
        correct, total = 0, 0
        valid_iterator = tqdm(valid_loader, desc='Validation loop') if verbose else valid_loader
        with torch.no_grad():
            for images, labels in valid_iterator:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, "
                  f"Valid loss: {valid_loss:.4f}, "
                  f"Valid acc: {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if save_checkpoint:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                if verbose:
                    print(f"  -> Saved new best model (valid_acc={valid_acc:.4f})")

    if save_losses:
        with open(LOSSES_PATH, 'w') as f:
            json.dump({'train': train_losses, 'valid': valid_losses}, f)

    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f)

    if verbose:
        print(f"\nDone. Best valid acc: {best_valid_acc:.4f}")

    return best_valid_acc


def train_with_best_params(num_epochs=10):
    with open(BEST_PARAMS_PATH, 'r') as f:
        best = json.load(f)

    params = best['params']

    print(f"Best params from Optuna (valid_acc={best['valid_accuracy']:.4f}):")
    for key, value in params.items():
        print(f"  {key}: {value}")

    return train(num_epochs=num_epochs, **params)


if __name__ == "__main__":
    if os.path.exists(BEST_PARAMS_PATH):
        train_with_best_params()
    else:
        train()