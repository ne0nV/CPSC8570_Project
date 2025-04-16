"""
robust_aggregation.py

Implements a robust aggregation mechanism using the Krum algorithm.
"""

import numpy as np

def krum_aggregation(updates, f=1):
    """
    Krum aggregation selects one update from the list that is most "central".

    Parameters:
      updates (list of np.array): List of client updates.
      f (int): Number of Byzantine (malicious) updates tolerated.
      
    Returns:
      np.array: Aggregated update.
    """
    num_updates = len(updates)
    scores = []
    
    # Compute pairwise distances between updates
    for i in range(num_updates):
        distances = []
        for j in range(num_updates):
            if i != j:
                distances.append(np.linalg.norm(updates[i] - updates[j]))
        distances = np.array(distances)
        # Sum the distances to the closest (num_updates - f - 2) updates
        sorted_distances = np.sort(distances)
        score = np.sum(sorted_distances[:max(num_updates - f - 2, 1)])
        scores.append(score)
    
    # Select the update with the minimal score
    best_index = int(np.argmin(scores))
    print(f"Krum selected update from client index {best_index} with score {scores[best_index]:.4f}")
    return updates[best_index]
