"""
cross_validation.py

Updated cross-validation function to evaluate the global model using torch.
"""

import torch

def validate_model(model, val_loader):
    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
