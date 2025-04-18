"""
cross_validation.py
Updated cross-validation function to evaluate the global model using torch.
"""
import torch
import numpy as np

def validate_model(model, val_loader):
    model.model.eval()
    correct = 0
    total = 0
    num_classes = 10  # CIFAR-10 or Fashion-MNIST have 10 classes
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # Update confusion matrix
            for i, j in zip(y.cpu().numpy(), predicted.cpu().numpy()):
                confusion[i, j] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    print("\nConfusion Matrix:")
    print(confusion)
    
    per_class_acc = np.diag(confusion) / np.sum(confusion, axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc:.4f}")
    
    # Check for targeted attack effect
    source_class = 4  # coat
    target_class = 6  # shirt
    misclass_rate = 0
    if confusion[source_class].sum() > 0:  # Avoid division by zero
        misclass_rate = confusion[source_class, target_class] / np.sum(confusion[source_class])
        print(f"\nMisclassification rate from class {source_class} to {target_class}: {misclass_rate:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'per_class_recall': per_class_acc,
        'source_class_recall': per_class_acc[source_class],
        'source_to_target_rate': misclass_rate
    }
    
    return metrics
