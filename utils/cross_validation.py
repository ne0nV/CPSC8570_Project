"""
cross_validation.py

Provides a simple cross-validation function to evaluate the global model.
"""

import numpy as np

def validate_model(model):
    """
    Simulates cross-validation on the model.

    Returns:
      float: A dummy validation score between 0.4 and 0.9.
    """
    return np.random.uniform(0.4, 0.9)
