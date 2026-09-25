import json
import os
import matplotlib.pyplot as plt

from config import LOSSES_PATH, PLOT_PATH


def plot_losses(losses_path=LOSSES_PATH, save_path=PLOT_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(losses_path, 'r') as f:
        data = json.load(f)

    plt.plot(data['train'], label='Training loss')
    plt.plot(data['valid'], label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.show()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    plot_losses()