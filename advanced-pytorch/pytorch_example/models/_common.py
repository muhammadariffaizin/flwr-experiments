"""Reusable model building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCNN(nn.Module):
    """Compact CNN for image classification."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(width * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TabularMLP(nn.Module):
    """MLP for tabular classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x)


class TextClassifier(nn.Module):
    """Embedding-based classifier for tokenized text."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = (embedded * mask).sum(dim=1) / lengths
        return self.fc2(F.relu(self.fc1(pooled)))


def get_int(config, key: str, default: int) -> int:
    """Read an integer model parameter from Flower run config."""
    if config is None:
        return default
    return int(config.get(key, default))
