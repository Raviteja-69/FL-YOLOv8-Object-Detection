# FL-YOLOv8-Object-Detection

This repository demonstrates **Federated Learning (FL)** applied to **YOLOv8 object detection**. The goal is to collaboratively train a robust object detection model from data distributed across multiple "clients," without sharing raw data.

---

## Project Structure

```
FederatedLearning/
├── core/                  # Shared utilities for aggregation, model adaptation, etc.
│   ├── aggregation.py
│   └── utils.py
├── FL_WithSequentialTraining/   # Sequential FL (clients train one after another)
│   ├── Main.py
│   └── train_client.py
├── FL_WithMultiprocessing/      # Multiprocessing FL (clients train in parallel)
│   ├── main.py
│   └── client.py
├── Evaluate_models/      # Scripts for evaluating trained models
├── partition_dataset.py  # Script to partition dataset among clients (IID, non-IID, custom)
├── partition_dataset_instructions.txt # Instructions for dataset partitioning
├── requirements.txt
├── README.md
└── ... (other files)
```

---

## Key Components

### 1. **Core Utilities (`core/`)**
- **`aggregation.py`**: Standardized federated averaging for model weights.
- **`utils.py`**: Utilities for adapting the YOLOv8 model head to the correct number of classes (dummy training step).

### 2. **Federated Learning Approaches**
- **Sequential Training (`FL_WithSequentialTraining/`)**: Clients train one after another in each round. Orchestrated by `Main.py`.
- **Multiprocessing Training (`FL_WithMultiprocessing/`)**: Clients train in parallel using Python multiprocessing. Orchestrated by `main.py`.
- Both approaches use the same core aggregation and model adaptation utilities for consistency and reproducibility.

### 3. **Dataset Partitioning**
- **`partition_dataset.py`**: Flexible script to split a YOLO-format dataset among any number of clients.
  - Supports IID (random), non-IID (by class), and custom proportions (per-class, per-client).
  - Generates correct folder structure and `data.yaml` for each client.
  - See `partition_dataset_instructions.txt` for detailed usage and examples.

### 4. **Model Evaluation**
- **`Evaluate_models/`**: Scripts for evaluating global or client models on validation/test sets.

---

## Training Workflow

1. **Dataset Preparation**
   - Use `partition_dataset.py` to split your dataset among clients as desired (IID, non-IID, custom).
   - Each client will have its own `train/` folder and `data.yaml`.

2. **Global Model Head Adaptation**
   - Before FL rounds, the server runs a dummy training step (using `core/utils.py`) to adapt the YOLOv8 model head to the correct number of classes.
   - This ensures all client and global models are compatible for aggregation.

3. **Federated Learning Rounds**
   - **Sequential:** Each client trains in turn, then the server aggregates their models.
   - **Multiprocessing:** Clients train in parallel, then the server aggregates their models.
   - Aggregation is performed using the standardized function in `core/aggregation.py`.

4. **Evaluation**
   - Use scripts in `Evaluate_models/` to assess model performance on validation or test sets.

---

## Example: Partitioning the Dataset

See `partition_dataset_README.txt` for full instructions. Example command:

```bash
python partition_dataset.py --dataset_root /path/to/your/dataset --num_clients 2 --split_type iid
```

---

## Notes
- The codebase is modular and ready for experiments with different data splits, aggregation strategies, and FL workflows.
- Model head adaptation is **critical** for correct aggregation—handled automatically by the core utilities.
- For custom non-IID splits, provide a YAML file with per-class, per-client proportions.
- All scripts are designed for reproducibility and easy extension.

---

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Partition your dataset: `python partition_dataset.py ...`
3. Run federated training: `python FL_WithSequentialTraining/Main.py` or `python FL_WithMultiprocessing/main.py`
4. Evaluate models: see `Evaluate_models/`

---
