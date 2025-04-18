"""
label_flipping.py
Implements label flipping attacks as described in "Data Poisoning Attacks 
Against Federated Learning Systems"
"""
import logging
import torch

logger = logging.getLogger(__name__)

def poison_client_data(client_loader, source_class, target_class):
    """
    Flips labels in a client's dataset from source_class to target_class.
    returns the modified DataLoader with poisoned labels
    """
    dataset = client_loader.dataset
    flipped = 0
    
    if hasattr(dataset, 'dataset'):
        subset = dataset
        parent_dataset = subset.dataset
        indices = subset.indices
        
        if hasattr(parent_dataset, 'targets'):
            targets = parent_dataset.targets
            
            # Convert to list if it's a tensor or other type
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
                parent_dataset.targets = targets
            elif not isinstance(targets, list):
                targets = list(targets)
                parent_dataset.targets = targets
            
            # Count flipped labels
            for idx in indices:
                if targets[idx] == source_class:
                    targets[idx] = target_class
                    flipped += 1
    else:
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
                dataset.targets = targets
            elif not isinstance(targets, list):
                targets = list(targets)
                dataset.targets = targets
            
            for i in range(len(targets)):
                if targets[i] == source_class:
                    targets[i] = target_class
                    flipped += 1
    
    print(f"Label flipping: changed {flipped} instances from class {source_class} to {target_class}")
    return client_loader
