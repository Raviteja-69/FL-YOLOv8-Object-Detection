# Federated Learning 

This folder contains the code for a Federated Learning (FL) setup where:

* **Clients train one after another** within each communication round.
* The server manages the global model, sending it to each client in turn.
* Each client updates the model with its local data and sends its changes back to the server.
* After all clients have trained, the server aggregates their updates to create a new global model for the next round.

* **`main.py`**:
    * This is the **server-side orchestrator**.
    * It manages the entire federated learning process, including rounds, sending models to clients, and initiating aggregation.
* **`train_client.py`**:
    * This file handles the **local training process for a single client**.
    * It loads the client's data, trains the model, and saves the updated weights.
* **`aggregate_weights.py`**:
    * This script is responsible for **averaging the model weights** received from all participating clients to create a new, aggregated global model.
