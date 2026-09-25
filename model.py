import json
import torch
import torch.nn as nn
import timm


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = self.base_model.num_features
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


def load_trained_model(checkpoint_path, classes_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(classes_path, 'r') as f:
        class_names = json.load(f)

    model = SimpleCardClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names, device