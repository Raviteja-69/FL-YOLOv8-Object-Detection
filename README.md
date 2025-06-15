# FL-YOLOv8-Object-Detection

This repository contains a project demonstrating **Federated Learning (FL)** applied to **YOLOv8 object detection**. Our goal is to train a robust object detection model collaboratively from data distributed across multiple "clients," without the clients ever sharing their raw data.

---

## Project Structure

This repository offers two main approaches for federated learning:

### 1. `FL_SequentialTraining/`
This folder holds the code for an FL setup where clients train **one after another (sequentially)** within each communication round.
* **`main.py`**: This is the **server-side orchestrator**. It manages the entire federated learning process, including rounds, sending models to clients, and initiating aggregation.
* **`train_client.py`**: This file handles the **local training process for a single client**. It loads the client's data, trains the model, and saves the updated weights.
* **`aggregate_weights.py`**: This script is responsible for **averaging the model weights** received from all participating clients to create a new, aggregated global model.

### 2. `FL_Multiprocessing/`
This folder contains code designed to enable **parallel client training** using multiprocessing. This approach aims to speed up federated learning rounds by allowing multiple clients to train simultaneously.

* **Note on Parallelism:** This setup truly benefits from **multiple GPUs or CPU cores**. On single-GPU environments like Google Colab's free tier, clients will still execute one after another, despite the multiprocessing structure.

---

## How It Works (Overall Concept)

1.  **Global Model Adaptation:** An initial YOLOv8 model is adapted to your custom dataset's number of classes on the server.
2.  **Client Training:** In each round, clients receive the current global model. They then train this model on their private local datasets for a few epochs.
3.  **Model Aggregation:** Clients send their updated model weights back to the server. The server aggregates these weights (typically by averaging them) to create an improved global model.
4.  **Iteration:** This process repeats for a specified number of rounds, iteratively improving the global model without direct data sharing.

---
