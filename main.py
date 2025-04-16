"""
main.py

Main simulation script for federated learning with enhanced defenses against data poisoning attacks.
"""

import numpy as np
import torch
from models.federated_model import FederatedModel
from defenses.robust_aggregation import krum_aggregation
from defenses.differential_privacy import add_dp_noise
from defenses.reputation_system import ReputationSystem
from defenses.monitoring import monitor_performance
from utils.cross_validation import validate_model
from attacks import label_flipping, backdoor_attack, data_injection

# Hyperparameters and simulation settings
NUM_CLIENTS = 20
NUM_ATTACKERS = 3  # number of malicious clients
NUM_ROUNDS = 10
NOISE_STD = 0.1  # for differential privacy
F = 1  # Number of Byzantine clients tolerated (for Krum)

# Create a federated model instance
global_model = FederatedModel()

# Initialize a reputation system for clients
reputation = ReputationSystem(num_clients=NUM_CLIENTS)

def simulate_local_update(client_id, model):
    """
    Simulate a local update.
    In a real scenario, this would train the model on local data.
    """
    # Create a dummy update vector (e.g., gradient) with the same shape as model parameters
    update = np.random.randn(*model.get_parameters().shape)
    return update

# Simulation loop
for round_num in range(NUM_ROUNDS):
    print(f"\n--- Round {round_num+1} ---")
    client_updates = []
    client_ids = list(range(NUM_CLIENTS))
    
    # Simulate local updates from all clients
    for client_id in client_ids:
        update = simulate_local_update(client_id, global_model)
        
        # Apply a poisoning attack if the client is malicious
        if client_id < NUM_ATTACKERS:
            if round_num % 3 == 0:
                update = label_flipping.poison_update(update)
                print(f"Client {client_id}: Label-flipping attack applied.")
            elif round_num % 3 == 1:
                update = backdoor_attack.poison_update(update)
                print(f"Client {client_id}: Backdoor attack applied.")
            else:
                update = data_injection.poison_update(update)
                print(f"Client {client_id}: Data injection attack applied.")
        
        client_updates.append((client_id, update))
    
    # Weight updates by reputation scores
    weighted_updates = []
    for client_id, update in client_updates:
        weight = reputation.get_trust(client_id)
        weighted_updates.append(weight * update)
    
    # Use robust aggregation (Krum) on weighted updates
    aggregated_update = krum_aggregation(weighted_updates, f=F)
    
    # Add differential privacy noise
    aggregated_update = add_dp_noise(aggregated_update, noise_std=NOISE_STD)
    
    # Update the global model with the aggregated update
    global_model.update_parameters(aggregated_update)
    
    # Validate global model performance using cross-validation
    validation_score = validate_model(global_model)
    print(f"Validation Score: {validation_score:.4f}")
    
    # Monitor performance and update reputations
    reputation.update(client_updates, validation_score)
    monitor_performance(round_num, validation_score)

print("\nSimulation complete.")
