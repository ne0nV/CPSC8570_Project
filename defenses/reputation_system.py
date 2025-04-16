"""
reputation_system.py

Implements a simple reputation system that assigns and updates trust scores for clients.
"""

class ReputationSystem:
    def __init__(self, num_clients):
        # Initialize all clients with a neutral trust score of 1.0
        self.trust_scores = {client_id: 1.0 for client_id in range(num_clients)}
    
    def get_trust(self, client_id):
        return self.trust_scores.get(client_id, 1.0)
    
    def update(self, client_updates, validation_score):
        """
        Update trust scores based on the validation performance.
        If the global performance is poor, decrease trust; otherwise, slightly increase.
        """
        for client_id, _ in client_updates:
            if validation_score < 0.5:
                self.trust_scores[client_id] *= 0.95
            else:
                self.trust_scores[client_id] *= 1.01
        
        # Normalize trust scores for stability
        total = sum(self.trust_scores.values())
        for key in self.trust_scores:
            self.trust_scores[key] /= total

    def __repr__(self):
        return str(self.trust_scores)
