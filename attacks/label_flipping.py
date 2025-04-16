"""
label_flipping.py

Implements a simple label-flipping attack.
"""

def poison_update(update):
    """
    Simulates a label-flipping attack by inverting the sign of the update.
    
    Parameters:
      update (np.array): Original update.
      
    Returns:
      np.array: Poisoned update.
    """
    return -update
