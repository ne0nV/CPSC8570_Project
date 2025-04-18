"""
label_flipping.py
Implements label flipping attacks as described in "Data Poisoning Attacks 
Against Federated Learning Systems"
"""
import logging
import torch

logger = logging.getLogger(__name__)

def poison_client_data(client_loader, source_class=0, target_class=2):
    """
    Flips labels in a client's dataset from source_class to target_class.
    returns the modified DataLoader with poisoned labels
    """
    dataset = client_loader.dataset
    flipped = 0
    
    # Get access to the underlying dataset if it's a subset
    if hasattr(dataset, 'dataset'):
        # It's a subset
        subset = dataset
        parent_dataset = subset.dataset
        indices = subset.indices
        
        # Access targets
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
        # Direct dataset access
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            
            # Convert to list if needed
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
                dataset.targets = targets
            elif not isinstance(targets, list):
                targets = list(targets)
                dataset.targets = targets
            
            # Flip labels
            flipped = 0
            for i in range(len(targets)):
                if targets[i] == source_class:
                    targets[i] = target_class
                    flipped += 1
    
    print(f"Label flipping: changed {flipped} instances from class {source_class} to {target_class}")
    return client_loader
