"""
main.py

Federated learning simulation with support for CIFAR-10 and Fashion-MNIST.
"""

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
from attacks import label_flipping, backdoor_attack, data_injection
import os
import pandas as pd
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    return parser.parse_args()

def load_dataset(name, num_clients):
    if name == "FashionMNIST":
        dataset_class = datasets.FashionMNIST
        input_size = 28 * 28
    elif name == "CIFAR10":
        dataset_class = datasets.CIFAR10
        input_size = 3 * 32 * 32
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = dataset_class(root='./data', train=True, download=True, transform=transform)
    val_data = dataset_class(root='./data', train=False, download=True, transform=transform)

    client_data = []
    data_per_client = len(train_data) // num_clients
    for i in range(num_clients):
        subset = torch.utils.data.Subset(train_data, range(i * data_per_client, (i + 1) * data_per_client))
        loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
        client_data.append(loader)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
    return client_data, val_loader, input_size

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    rounds = config['rounds']
    dp_enabled = config['defenses']['differential_privacy']
    dp_std = config['defenses'].get('dp_std', 0.1)

    client_loaders, val_loader, input_size = load_dataset(config['dataset'], num_clients)
    global_model = FederatedModel(input_dim=input_size, output_dim=10)
    reputation = ReputationSystem(num_clients)
    log = []

    for rnd in range(rounds):
        print(f"\n--- Round {rnd+1} ---")
        updates = []
        for cid, loader in enumerate(client_loaders):
        # Apply attacks to the data before training for malicious clients
            current_loader = loader
            if cid < num_attackers:
                attack_type = config['attacks'][0]['type']
                if attack_type == 'label_flipping':
                    source_class = config['attacks'][0].get('source_class', 0)
                    target_class = config['attacks'][0].get('target_class', 2)
                    # Modify the client's data before training
                    current_loader = label_flipping.poison_client_data(
                        loader, source_class=source_class, target_class=target_class)
                    print(f"Client {cid}: {attack_type} attack applied (flipping class {source_class} to {target_class}).")
                elif attack_type == 'backdoor':
                    current_loader = backdoor_attack.poison_client_data(loader)
                    print(f"Client {cid}: {attack_type} attack applied.")
                elif attack_type == 'data_injection':
                    current_loader = data_injection.poison_client_data(loader)
                    print(f"Client {cid}: {attack_type} attack applied.")

            # Train using the possibly poisoned data
            update = global_model.train_on_client(current_loader)
            weight = reputation.get_trust(cid)
            updates.append(weight * update)


        #aggregated = krum_aggregation(updates, f=1) replacing this to be simpler for testing purposes of attack
        aggregated = sum(updates) / len(updates)  # Simple averaging

        if dp_enabled:
            aggregated = add_dp_noise(aggregated, dp_std)

        global_model.update_parameters(aggregated)
        metrics = validate_model(global_model, val_loader)
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Source class (0) recall: {metrics['source_class_recall']:.4f}")
        print(f"Misclassification rate from source to target: {metrics['source_to_target_rate']:.4f}")
        log.append({'round': rnd + 1, 'accuracy': metrics['accuracy']})
        reputation.update([(cid, updates[cid]) for cid in range(num_clients)], metrics['accuracy'])
        monitor_performance(rnd, metrics['accuracy'])

    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    log_path = os.path.join(config['logging']['output_dir'], 'metrics.csv')
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"Saved metrics to {log_path}")

if __name__ == '__main__':
    main()
