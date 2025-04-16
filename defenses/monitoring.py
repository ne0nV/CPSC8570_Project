"""
monitoring.py

Provides continuous monitoring of the federated model performance.
"""

def monitor_performance(round_num, validation_score):
    """
    Logs and monitors the performance of the global model.
    
    Parameters:
      round_num (int): Current training round.
      validation_score (float): Validation performance metric.
    """
    threshold = 0.4
    if validation_score < threshold:
        print(f"WARNING: Validation score dropped below {threshold} at round {round_num+1}.")
    else:
        print(f"Round {round_num+1} performance is stable.")
