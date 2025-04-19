"""
label_flipping.py
"""
import torch
import copy

def poison_client_data(client_loader, source_class=0, target_class=2):
    """
    Creates a poisoned DataLoader that flips labels.
    Returns a new DataLoader with poisoned labels
    """
    original_dataset = client_loader.dataset
    batch_size = client_loader.batch_size
    
    flipped = 0
    indices_to_flip = []
    
    for i in range(len(original_dataset)):
        try:
            item = original_dataset[i]
            
            if isinstance(item, tuple) and len(item) == 2:
                _, label = item
                if label == source_class:
                    flipped += 1
                    indices_to_flip.append(i)
        except Exception as e:
            # Skip any problematic entries
            continue
    
    print(f"Label flipping: found {flipped} instances from class {source_class} to flip to {target_class}")
    
    class PoisonedDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.original_dataset = original_dataset
            self.indices_to_flip = set(indices_to_flip)
            
        def __len__(self):
            return len(self.original_dataset)
            
        def __getitem__(self, idx):
            item = self.original_dataset[idx]
            
            if isinstance(item, tuple) and len(item) == 2:
                features, label = item
                
                if idx in self.indices_to_flip:
                    return features, target_class
                else:
                    return features, label
            else:
                return item
    
    poisoned_dataset = PoisonedDataset()
    
    poisoned_loader = torch.utils.data.DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return poisoned_loader
