"""
federated_model.py

CNN model adaptable to both Fashion-MNIST and CIFAR-10 datasets.
"""

import torch
import torch.nn as nn
import numpy as np

class FederatedModel:
    def __init__(self, input_channels=1, output_dim=10):
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Output 128x4x4 feature map
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def train_on_client(self, data_loader, epochs=1):
        self.model.train()
        initial = self.get_parameters()
        for _ in range(epochs):
            for x, y in data_loader:
                self.optimizer.zero_grad()
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()
        updated = self.get_parameters()
        return updated - initial

    def get_parameters(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])

    def update_parameters(self, update):
        pointer = 0
        for param in self.model.parameters():
            numel = param.data.numel()
            param.data.add_(torch.tensor(update[pointer:pointer+numel].reshape(param.data.shape), dtype=torch.float32))
            pointer += numel
