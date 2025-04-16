# Project Tasks

## `main.py`

**Current:**
- Simulates random updates  
- Orchestrates rounds with dummy updates  
- No real training or data  

**To-do:**
- Use real local datasets per client (e.g., MNIST, CIFAR10)  
- Replace `<simulate_local_update>` with actual local training  
- Implement model serialization across rounds  
- Allow CLI args for hyperparameter tuning? (`argparse`)  

---

## `models/federated_model.py`

**Current:**
- Simple linear model with random input/output  

**To-do:**
- Include architecture for images (CNNs)  
- Add `.train_on_data(data)` method  
- Build PyTorch `nn.Module` wrappers for `FedModel` abstraction  

---

## `attacks/`

**Current:**
- Label flipping, data injection, backdoor attacks as toy examples  

**To-do:**
- Use dataset-aware attacks (e.g., label flipping on actual models — CIFAR, MNIST, etc.)  
- Allow configurable attack type, target class, ratio of attackers  

---

## `defenses/`

**Current:**
- Krum aggregation  
- Differential privacy (Gaussian noise)  
- Basic reputation score  
- Simple performance threshold monitor  

**To-do:**
- Enhance Diff Priv with tunable E-diff privacy and clipping norms  
- Extend reputation system:
  - Track historical behavior  
  - Use exponential decay  
- Add dynamic defenses: remove/penalize clients during training  
- Add anomaly detection using clustering (e.g., DBSCAN on updates)  

---

## `utils/`

**Current:**
- Cross-validation stub  
- Regularization stub  

**To-do:**
- Load real validation datasets per round  
- Implement real-time metrics logging (accuracy, loss, etc.)  
- Add metrics tracker (such as Weights & Biases or CSV logger)  
- Implement per-client local loss tracking to detect bad actors  

---

## `experiments/`

**Current:**
- Runs a single simulation via script  

**To-do:**
- Support multiple experiment configurations (YAML/JSON)  
- Parameter sweeps: attacker ratio, noise levels, defense strategy  
- Export logs/metrics to CSV or plotting dashboards  
- Add automatic evaluation scripts to compute:
  - Attack success rate  
  - Global model accuracy  
  - Defense overhead (computation/time)  

---

## Additional Modules

### `data/`
- Loaders for federated datasets (e.g., CIFAR-10, FMNIST)  

### `config/`
- YAML or JSON configs for experiment reproduction  

### `visuals/`
- Plotting utilities for accuracy, trust scores, gradient norms  
