"""
data_injection.py

Implements a data injection attack by adding adversarial noise to the update.
"""

import numpy as np

def poison_update(update):
    """
    Simulates a data injection attack by adding adversarial noise.
    
    Parameters:
      update (np.array): Original update.
      
    Returns:
      np.array: Update with injected adversarial noise.
    """
    adversarial_noise = np.random.normal(0, 0.5, size=update.shape)
    return update + adversarial_noise
