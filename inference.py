import random
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from model import load_trained_model
from dataset import get_transform, download_dataset

from config import CHECKPOINT_PATH, CLASSES_PATH


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def show_predict(image_dir_pattern, n=5, seed=None):
    model, class_names, device = load_trained_model(CHECKPOINT_PATH, CLASSES_PATH)
    transform = get_transform()

    test_images = glob(image_dir_pattern)
    if not test_images:
        print(f"No images found for pattern: {image_dir_pattern}")
        return

    if seed is not None:
        random.seed(seed)

    test_examples = random.sample(test_images, min(n, len(test_images)))

    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transform)
        probabilities = predict(model, image_tensor, device)
        visualize_predictions(original_image, probabilities, class_names)


if __name__ == "__main__":
    path = download_dataset()
    show_predict(os.path.join(path, 'test', '*', '*'))