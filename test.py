import torch
from dataset import get_dataloaders
from model import load_trained_model

from config import CHECKPOINT_PATH, CLASSES_PATH

_, _, test_loader, _ = get_dataloaders()

model, class_names, device = load_trained_model(CHECKPOINT_PATH, CLASSES_PATH)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")