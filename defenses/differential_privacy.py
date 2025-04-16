"""
differential_privacy.py

Adds differential privacy by injecting Gaussian noise to updates.
"""

import numpy as np

def add_dp_noise(update, noise_std=0.1):
    """
    Adds Gaussian noise to the update for differential privacy.

    Parameters:
      update (np.array): Aggregated update.
      noise_std (float): Standard deviation of the Gaussian noise.
      
    Returns:
      np.array: Noisy update.
    """
    noise = np.random.normal(0, noise_std, size=update.shape)
    return update + noise
