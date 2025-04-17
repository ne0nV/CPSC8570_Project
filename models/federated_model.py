"""
federated_model.py

MLP model adaptable to both Fashion-MNIST and CIFAR-10 datasets.
"""

import torch
import torch.nn as nn
import numpy as np

class FederatedModel:
    def __init__(self, input_dim, output_dim=10):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

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
