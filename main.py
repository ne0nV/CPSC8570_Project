import argparse
import yaml
import numpy as np
import torch
from utils.cross_validation import validate_model
from models.federated_model import FederatedModel
from defenses.robust_aggregation import krum_aggregation
from defenses.differential_privacy import add_dp_noise
from defenses.reputation_system import ReputationSystem
from defenses.monitoring import monitor_performance
from attacks import label_flipping
import os
import pandas as pd
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    return parser.parse_args()

def load_dataset(name, num_clients):
    """
    Load the dataset but don't distribute it to clients yet.
    We will distribute it differently in each round.
    """
    if name == "FashionMNIST":
        dataset_class = datasets.FashionMNIST
        input_channels = 1
    elif name == "CIFAR10":
        dataset_class = datasets.CIFAR10
        input_channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    val_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    return train_dataset, val_loader, input_channels

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    rounds = config['rounds']
    dp_enabled = config['defenses']['differential_privacy']
    dp_std = config['defenses'].get('dp_std', 0.1)
    
    # Load the full dataset but don't distribute yet
    train_dataset, val_loader, input_channels = load_dataset(config['dataset'], num_clients)
    
    global_model = FederatedModel(input_channels=input_channels, output_dim=10)
    reputation = ReputationSystem(num_clients)
    log = []
    
    for rnd in range(rounds):
        print(f"\n--- Round {rnd+1} ---")
        
        # Create a fresh copy of the training dataset
        # Distribute data differently in each round
        total_size = len(train_dataset)
        indices = torch.randperm(total_size).tolist()
        data_per_client = total_size // num_clients
        
        client_loaders = []
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = (i + 1) * data_per_client if i < num_clients - 1 else total_size
            
            client_indices = indices[start_idx:end_idx]
            subset = torch.utils.data.Subset(train_dataset, client_indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
            client_loaders.append(loader)
        
        updates = []
        for cid, loader in enumerate(client_loaders):
            # Apply attacks to the data before training for malicious clients
            if cid < num_attackers:
                attack_type = config['attacks'][0]['type']
                if attack_type == 'label_flipping':
                    source_class = config['attacks'][0].get('source_class')
                    target_class = config['attacks'][0].get('target_class')
                    
                    # Simple but effective label flipping implementation
                    subset = loader.dataset
                    dataset = subset.dataset
                    indices = subset.indices
                    
                    # Count flips for logging
                    flipped = 0
                    
                    # Create a temporary dataset with flipped labels for training
                    targets_copy = dataset.targets.clone() if isinstance(dataset.targets, torch.Tensor) else dataset.targets.copy()
                    original_targets = dataset.targets
                    dataset.targets = targets_copy  # Use the copy for this round
                    
                    # Apply flipping
                    for idx in indices:
                        if dataset.targets[idx] == source_class:
                            dataset.targets[idx] = target_class
                            flipped += 1
                    
                    print(f"Label flipping: changed {flipped} instances from class {source_class} to {target_class}")
                    print(f"Client {cid}: {attack_type} attack applied (flipping class {source_class} to {target_class}).")
                    
                    # Train with poisoned data
                    update = global_model.train_on_client(loader)
                    
                    # Restore original targets for the next round
                    dataset.targets = original_targets
                else:
                    # Handle other attack types
                    update = global_model.train_on_client(loader)
            else:
                update = global_model.train_on_client(loader)
            
            weight = reputation.get_trust(cid)
            updates.append(weight * update)
        
        # Simple averaging instead of Krum
        aggregated = sum(updates) / len(updates)
        
        if dp_enabled:
            aggregated = add_dp_noise(aggregated, dp_std)
        
        global_model.update_parameters(aggregated)
        
        metrics = validate_model(global_model, val_loader)
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Source class recall: {metrics['source_class_recall']:.4f}")
        print(f"Misclassification rate from source to target: {metrics['source_to_target_rate']:.4f}")
        
        log.append({
            'round': rnd + 1, 
            'accuracy': metrics['accuracy'],
            'source_recall': metrics['source_class_recall'],
            'misclassification_rate': metrics['source_to_target_rate']
        })
        
        reputation.update([(cid, updates[cid]) for cid in range(num_clients)], metrics['accuracy'])
        
        if metrics['accuracy'] < 0.4:
            print(f"WARNING: Validation score dropped below 0.4 at round {rnd+1}.")
    
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    log_path = os.path.join(config['logging']['output_dir'], 'metrics.csv')
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"Saved metrics to {log_path}")

if __name__ == '__main__':
    main()
