# Federated Learning (Sequential Training)

This folder contains the code for a **Federated Learning (FL) setup with sequential client training**:

- **Clients train one after another** within each communication round.
- The server manages the global model, sending it to each client in turn.
- Each client updates the model with its local data and sends its changes back to the server.
- After all clients have trained, the server aggregates their updates to create a new global model for the next round.

## Key Files

- **`Main.py`**:
    - The **server-side orchestrator** for sequential FL.
    - Manages FL rounds, sends models to clients, and performs aggregation using the shared core utility.
    - Handles model head adaptation (via `core/utils.py`) to ensure compatibility for aggregation.
- **`train_client.py`**:
    - Handles the **local training process for a single client**.
    - Loads the client's data, trains the model, and saves the updated weights.

## Shared Utilities
- **Model aggregation and head adaptation** are now handled by the shared `core/` utilities:
    - `core/aggregation.py`: Standardized federated averaging.
    - `core/utils.py`: Dummy training for model head adaptation.

## Dataset Partitioning
- Use the top-level `partition_dataset.py` script to split your dataset among clients (IID, non-IID, or custom splits).
- See `../partition_dataset_README.txt` for detailed instructions.

## Notes
- This folder is for **sequential FL**. For multiprocessing/parallel FL, see `../FL_WithMultiprocessing/`.
- For overall project structure, advanced features, and evaluation, see the main `../README.md`.
