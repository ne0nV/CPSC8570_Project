"""
backdoor_attack.py

Simulates a backdoor attack by injecting a specific pattern into the update.
"""

import numpy as np

def poison_update(update):
    """
    Injects a backdoor pattern into the update.
    
    Parameters:
      update (np.array): Original update.
      
    Returns:
      np.array: Modified update with a backdoor pattern.
    """
    backdoor_pattern = 0.1 * np.ones_like(update)
    return update + backdoor_pattern
