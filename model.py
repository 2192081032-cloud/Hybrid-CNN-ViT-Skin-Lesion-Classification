import torch
import torch.nn as nn
from torchvision import models


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # ResNet18 convolutional feature extractor
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Convert CNN features into 256-dimensional tokens
        self.embedding = nn.Linear(512, 256)

        # Positional embeddings for 49 tokens
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 49, 256)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Seven-class classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)

        tokens = features.flatten(2).transpose(1, 2)

        tokens = self.embedding(tokens)

        tokens = tokens + self.pos_embedding

        tokens = self.transformer(tokens)

        pooled = tokens.mean(dim=1)

        output = self.classifier(pooled)

        return output