import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import kagglehub


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])


def download_dataset():
    return kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")


def get_dataloaders(batch_size=32):
    path = download_dataset()

    transform = get_transform()

    train_dataset = PlayingCardDataset(os.path.join(path, 'train'), transform=transform)
    valid_dataset = PlayingCardDataset(os.path.join(path, 'valid'), transform=transform)
    test_dataset = PlayingCardDataset(os.path.join(path, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, train_dataset.classes