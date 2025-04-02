# CPSC8570_Project
FL Project improvements for CPSC 8570

# DataPoisoning_FL_Improved

This repository is an enhanced implementation of federated learning under data poisoning attacks. It builds on prior work by incorporating these improvements:

- **Diversified Attack Scenarios:** In addition to label-flipping, includes backdoor and data injection attacks.
- **Robust Aggregation:** Implements Byzantine-robust aggregation using the Krum algorithm.
- **Differential Privacy:** Adds Gaussian noise to aggregated updates.
- **Regularization & Cross-Validation:** Uses regularization and cross-validation to mitigate overfitting and detect anomalies.
- **Participant Reputation System:** Weights client updates by trust scores.
- **Continuous Monitoring:** Tracks model performance in real time.

## Setup

1. **Install Dependencies:**
   `pip install -r requirements.txt`

2. **Run the Main Simulation:**
   `python main.py`

3. **Run Experiments:**
   `python experiments/run_experiment.py`

## Notes

**Modularity and Extensibility:**
Each module is designed to be self-contained. You can replace or extend any part (for example, by adding new attack types or more advanced robust aggregation methods).

**Experimentation:**
The experiments/run_experiment.py file serves as an entry point for running parameter sweeps or logging detailed results.
