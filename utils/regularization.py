"""
regularization.py

Implements regularization techniques to prevent overfitting to poisoned data.
"""

def apply_l2_regularization(update, lambda_reg=0.01):
    """
    Applies L2 regularization to the update.

    Parameters:
      update (np.array): The model update (e.g., gradient).
      lambda_reg (float): Regularization coefficient.
      
    Returns:
      np.array: Regularized update.
    """
    return update - lambda_reg * update
