import os
from ultralytics import YOLO
import torch
import torch.nn as nn
from copy import deepcopy
from multiprocessing import Pool

# Server Configuration
NUM_ROUNDS = 5
EPOCHS_PER_ROUND = 30  # This is for actual client training
NUM_CLIENTS_TO_SELECT_PER_ROUND = 2

CLIENT_MODEL_DIR = 'client_models'
GLOBAL_MODEL_CHECKPOINTS_DIR = 'global_model_checkpoints'

# Ensure directories exist
os.makedirs(CLIENT_MODEL_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_CHECKPOINTS_DIR, exist_ok=True)

# Initial global model (yolov8n.pt for the first round)
initial_global_model_path = 'yolov8n.pt'
NUM_CUSTOM_CLASSES = 4  # Define your custom number of classes here explicitly

# Client Configuration (Simulated)
CLIENT_DATA_CONFIGS = {
    'client1': 'ppekit/client1/data.yaml',  # Ensure this has nc: 4
    'client2': 'ppekit/client2/data.yaml',  # Ensure this has nc: 4
}


# Federated Averaging Function (FedAvg)
def fed_avg(global_model, client_models_state_dicts):
    global_weights = deepcopy(global_model.state_dict())
    aggregated_weights = {key: torch.zeros_like(value) for key, value in global_weights.items()}

    for client_sd in client_models_state_dicts:
        for key in aggregated_weights.keys():
            if key not in client_sd:
                print(f"Error: Key '{key}' missing in client state_dict!")
                continue

            if aggregated_weights[key].shape != client_sd[key].shape:
                # This should ideally not trigger
                print(f"Shape Mismatch for key: '{key}'")
                print(f"  Global model shape: {aggregated_weights[key].shape}")
                print(f"  Client model shape: {client_sd[key].shape}")

            aggregated_weights[key] += client_sd[key]

    for key in aggregated_weights.keys():
        aggregated_weights[key] = aggregated_weights[key] / len(client_models_state_dicts)

    global_model.load_state_dict(aggregated_weights)
    return global_model


# Client Training Function
def client_train_task(args):
    client_id, global_model_path, data_yaml, epochs, output_dir, current_round = args

    print(f"  Client {client_id}: Starting local training (Round {current_round + 1})...")

    client_model = YOLO(global_model_path)  # Clients load the global model as a .pt file

    client_run_name = f'{client_id}_round_{current_round}'
    client_output_path = os.path.join(output_dir, client_run_name)

    client_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        device='cpu',
        project=output_dir,
        name=client_run_name,
        verbose=False,
        save_period=0,
        save_txt=False,
        save_conf=False,
        plots=False
    )

    best_client_model_path = os.path.join(client_output_path, 'weights', 'best.pt')
    trained_model = YOLO(best_client_model_path)

    print(f"  Client {client_id}: Local training finished. Model state_dict prepared.")

    return trained_model.state_dict()


# Federated Learning Server Loop
if __name__ == "__main__":
    # --- CRITICAL FIX FOR GLOBAL MODEL INITIALIZATION ---
    # 1. Load the pre-trained YOLOv8n model.
    global_model = YOLO(initial_global_model_path)

    # 2. Force the global_model's Detect head to adapt to NUM_CUSTOM_CLASSES (4).
    #    The most reliable way Ultralytics does this is by running a training/validation step
    #    with a data.yaml that has the correct 'nc'.
    #    We'll use one of the client's data.yaml files for this, and run a dummy training round.

    print(f"Server: Forcing global model head adaptation to {NUM_CUSTOM_CLASSES} classes...")
    DUMMY_DATA_YAML_FOR_ADAPTATION = CLIENT_DATA_CONFIGS['client1']  # Use any client's data.yaml

    # Run a single dummy epoch training to force head adaptation.
    # We set minimal parameters to make it fast and not actually train much.
    global_model.train(
        data=DUMMY_DATA_YAML_FOR_ADAPTATION,
        epochs=1,  # Just 1 epoch to force adaptation
        imgsz=640,
        batch=1,  # Minimal batch size to reduce resource usage
        device='cpu',
        project='_temp_global_adapter',  # Temporary project folder
        name='_global_head_adapter_run',  # Temporary run name
        exist_ok=True,  # Allow overwriting temp folder
        save=False,  # Don't save weights from this dummy run
        plots=False,
        verbose=False  # Suppress verbose output
    )

    # After this dummy training, the global_model's head should be correctly adapted to number of classes of your custom dataset.
    # Clean up the temporary training directory created by this dummy run.
    import shutil

    temp_dir = os.path.join('_temp_global_adapter', '_global_head_adapter_run')
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Server: Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Server: Could not remove temporary directory {temp_dir}: {e}")

    # Verify the number of classes the global model now has (optional, but good for confidence)
    if hasattr(global_model.model, 'model') and len(global_model.model.model) > 0:
        if hasattr(global_model.model.model[-1], 'nc'):
            print(f"Global model's Detect head now configured for {global_model.model.model[-1].nc} classes.")
        else:
            print("Global model's last layer does not have a direct 'nc' attribute.")
    else:
        print("Could not determine global model's head classes directly.")
    # --- END CRITICAL FIX ---

    for current_round in range(NUM_ROUNDS):
        print(f"\n===== Federated Learning Round {current_round + 1}/{NUM_ROUNDS} =====")

        global_model_filename = f'global_model_r{current_round}.pt'
        current_global_model_path = os.path.join(GLOBAL_MODEL_CHECKPOINTS_DIR, global_model_filename)
        global_model.save(current_global_model_path)
        print(f"Server: Global model saved for round {current_round + 1} at {current_global_model_path}")

        client_tasks = []
        selected_client_ids = list(CLIENT_DATA_CONFIGS.keys())

        for client_id in selected_client_ids:
            data_yaml = CLIENT_DATA_CONFIGS[client_id]
            client_tasks.append(
                (client_id, current_global_model_path, data_yaml, EPOCHS_PER_ROUND, CLIENT_MODEL_DIR, current_round))

        client_models_state_dicts_for_aggregation = []

        with Pool(processes=os.cpu_count()) as pool:
            print(f"Server: Starting parallel training for {len(client_tasks)} clients...")
            results = pool.map(client_train_task, client_tasks)
            client_models_state_dicts_for_aggregation.extend(results)

        if client_models_state_dicts_for_aggregation:
            print(f"Server: Aggregating models from {len(client_models_state_dicts_for_aggregation)} clients...")
            global_model = fed_avg(global_model, client_models_state_dicts_for_aggregation)
            print("Server: Aggregation complete.")
        else:
            print("Server: No client models received for aggregation.")

    final_global_model_path = os.path.join(GLOBAL_MODEL_CHECKPOINTS_DIR, f'global_model_r{NUM_ROUNDS}.pt')
    global_model.save(final_global_model_path)
    print(f"\nFederated Learning training finished. Final model saved to {final_global_model_path}")

    # Evaluation Example (after training is complete)
    # VALIDATION_DATA_YAML = 'ppekit/global_valid/data.yaml'
    # print(f"\n--- Evaluating Final Model: {final_global_model_path} ---")
    # final_model_for_eval = YOLO(final_global_model_path)
    # final_model_for_eval.val(data=VALIDATION_DATA_YAML, imgsz=640, conf=0.25, iou=0.7, device='cpu')
