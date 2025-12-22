import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalCNN(nn.Module):
    def __init__(self, out_features=4):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ClassicalClassifier(nn.Module):
    """
    Wrapper for the ClassicalCNN to act as a full classifier.
    Structure: Input -> ClassicalCNN -> Features (4) -> Linear(10) -> Output
    """
    def __init__(self, cnn_extractor, n_classes=10):
        super().__init__()
        self.cnn = cnn_extractor
        # This linear layer takes the weak feature vector (size 4) and maps to classes
        self.classifier = nn.Linear(self.cnn.fc2.out_features, n_classes)

    def forward(self, x):
        features = self.cnn(x)
        out = self.classifier(features)
        return out, features

