# Project Tasks

## `main.py`

**Current:**
- Simulates random updates  
- Orchestrates rounds with dummy updates  
- No real training or data  

**To-do:**
- <s>Use real local datasets per client (e.g., MNIST, CIFAR10)</s>  
- <s>Replace `<simulate_local_update>` with actual local training</s>  
- Implement model serialization across rounds  
- <s>Allow CLI args for hyperparameter tuning? (`argparse`)</s>  

---

## `models/federated_model.py`

**Current:**
- Simple linear model with random input/output  

**To-do:**
- <s>Include architecture for images (CNNs)</s>  
- <s>Add `.train_on_data(data)` method</s>  
- <s>Build PyTorch `nn.Module` wrappers for `FedModel` abstraction</s>  

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
- <s>Support multiple experiment configurations (YAML/JSON)</s>  
- Parameter sweeps: attacker ratio, noise levels, defense strategy  
- <s>Export logs/metrics to CSV or plotting dashboards</s>  
- Add automatic evaluation scripts to compute:
  - Attack success rate  
  - Global model accuracy  
  - Defense overhead (computation/time)  

---

## Additional Modules

### `data/`
- Loaders for federated datasets (e.g., CIFAR-10, FMNIST)  

### <s>`config/`</s>
- <s>YAML or JSON configs for experiment reproduction</s>  

### `visuals/`
- Plotting utilities for accuracy, trust scores, gradient norms  
