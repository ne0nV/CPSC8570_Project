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

2. **Configure Experiment:**
   Open and edit the config file:
   `config/base_experiment.yaml`

   Example:
   <br/>
   ```
   experiment_name: "baseline_fl_demo"
   num_clients: 20
   num_attackers: 4
   rounds: 15
   dataset: "CIFAR10"         # or "FashionMNIST"
   model: "SimpleMLP"
   attacks:
     - type: "label_flipping"
       target_label: 7
   defenses:
     aggregation: "krum"
     differential_privacy: true
     dp_std: 0.1
     trust_system: true
   logging:
     output_dir: "logs/"
   ```

4. **Run the Experiment:**
   From the project root:
   `python main.py --config config/base_experiment.yaml`

   -OR-

   Use the Experiment Runner:
   `python experiments/run_experiment.py --config config/base_experiment.yaml`

6. **View Results:**
   A CSV log of round-wise accuracy will be saved to:
   `logs/metrics.csv`

## Notes

**Modularity and Extensibility:**
Each module is designed to be self-contained. You can replace or extend any part (for example, by adding new attack types or more advanced robust aggregation methods).

**Experimentation:**
The experiments/run_experiment.py file serves as an entry point for running parameter sweeps or logging detailed results.
