"""
federated_model.py

Defines a simple federated learning model.
For demonstration, a linear model using PyTorch is implemented.
"""

import torch
import torch.nn as nn
import numpy as np

class FederatedModel:
    def __init__(self, input_dim=10, output_dim=1):
        self.model = nn.Linear(input_dim, output_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def get_parameters(self):
        """
        Flattens model parameters into a single numpy array.
        """
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def update_parameters(self, update):
        """
        Applies the aggregated update to the model parameters.
        """
        with torch.no_grad():
            pointer = 0
            for param in self.model.parameters():
                num_params = param.data.numel()
                update_slice = update[pointer:pointer+num_params]
                update_tensor = torch.tensor(update_slice.reshape(param.data.shape), dtype=torch.float32)
                param.add_(update_tensor)
                pointer += num_params
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Quick test of the model functionality
    model = FederatedModel()
    dummy_input = torch.randn(1, 10)
    output = model.forward(dummy_input)
    print("Model output:", output)
