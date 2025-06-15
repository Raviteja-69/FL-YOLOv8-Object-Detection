import os
import subprocess
import torch
from ultralytics import YOLO
import shutil # For cleaning up temporary directories

# Configuration
NUM_ROUNDS = 3
# Ensure these paths are correct for your dataset in Google Drive
CLIENTS = [
    {'data_yaml': '/content/drive/MyDrive/Datasets/ppekit/client1/data.yaml', 'model_name': 'client1'},
    {'data_yaml': '/content/drive/MyDrive/Datasets/ppekit/client2/data.yaml', 'model_name': 'client2'},
]

# Define base directory for all outputs (models, checkpoints) in Google Drive
# Make sure you create this folder in your Google Drive: e.g., 'MyDrive/FL_YOLOv8_Project_Outputs'
FL_PROJECT_OUTPUTS_DIR = '/content/drive/MyDrive/FL_YOLOv8_Project_Outputs' # Unified output directory

GLOBAL_MODEL_SAVE_DIR = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'global_model_checkpoints')
os.makedirs(GLOBAL_MODEL_SAVE_DIR, exist_ok=True) # Ensure directory exists
INITIAL_GLOBAL_MODEL_PATH = os.path.join(GLOBAL_MODEL_SAVE_DIR, 'global_model_r0.pt')

EPOCHS_PER_ROUND = 1
NUM_CLASSES = 4 # Make sure this matches 'nc' in your data.yaml files

# Determine number of workers for client data loaders
# Use half of CPU cores, or adjust based on your needs/memory.
# For small datasets, 0 or 1 might be fine too.
NUM_WORKERS_FOR_CLIENT_DATALOADERS = os.cpu_count() // 2
if NUM_WORKERS_FOR_CLIENT_DATALOADERS == 0: # Ensure at least 1 worker if cpu_count is 1
    NUM_WORKERS_FOR_CLIENT_DATALOADERS = 1


def train_clients(round_num):
    print(f"\n=== Training Round {round_num + 1} ===")
    weights_paths = []

    # Define the current global model path
    current_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num}.pt')

    for client in CLIENTS:
        save_name = f"{client['model_name']}_round{round_num + 1}"

        # Determine starting weights for this round
        if round_num == 0:
            # For the very first round, use the ADAPTED global_model_r0.pt
            starting_weights = INITIAL_GLOBAL_MODEL_PATH
        else:
            # For subsequent rounds, use the *full checkpoint* of the aggregated global model
            starting_weights = current_global_model_path

        print(f"  Client {client['model_name']} starting with weights: {starting_weights}")

        subprocess.run([
            'python', 'FL_WithSequentialTraining/train_client.py',
            '--data', client['data_yaml'],
            '--save_name', save_name,
            '--epochs', str(EPOCHS_PER_ROUND),
            '--weights', starting_weights,
            '--device', 'cuda',
            '--workers', str(NUM_WORKERS_FOR_CLIENT_DATALOADERS)
        ], check=True)

        # Retrieve the path to the best.pt saved by ultralytics train command
        client_run_dir = os.path.join('runs', 'detect', save_name)
        best_pt_path = os.path.join(client_run_dir, 'weights', 'best.pt')

        if not os.path.exists(best_pt_path):
            raise FileNotFoundError(f"Client {client['model_name']} trained model not found at {best_pt_path}")

        weights_paths.append(best_pt_path)

        # if os.path.exists(client_run_dir):
        #     try:
        #         shutil.rmtree(client_run_dir)
        #         print(f"  Client {client['model_name']}: Cleaned up temporary run directory: {client_run_dir}")
        #     except Exception as e:
        #         print(f"  Client {client['model_name']}: Could not remove temporary directory {client_run_dir}: {e}")

    return weights_paths


def aggregate(weights_paths, round_num):
    # Import your aggregation script
    import aggregate_weights

    print(f"\n=== Aggregating Models for Round {round_num + 1} ===")

    # Average the weights using your aggregate_weights script's function
    # It will re-load weights from paths within that function.
    avg_state_dict = aggregate_weights.average_weights(weights_paths)

    # 3. Create a new global model with the correct structure (nc=4)
    # The best way is to load one of the *trained client models*
    # (which has the correct 4-class structure after initial adaptation)
    # and then update its weights with the averaged state_dict.
    base_model_for_aggregation = YOLO(weights_paths[0]) # Load a model with the correct head structure

    # Apply the averaged state_dict to this base model
    base_model_for_aggregation.model.load_state_dict(avg_state_dict)

    # Verify the nc of the aggregated model
    try:
        final_detection_layer_nc = base_model_for_aggregation.model.model[-1].nc
        print(f"Aggregated Model NC: {final_detection_layer_nc}")
        if final_detection_layer_nc != NUM_CLASSES:
            print(
                f"WARNING: Aggregated model nc is {final_detection_layer_nc}, expected {NUM_CLASSES}. "
                "Ensure your data.yaml is correctly configured for {NUM_CLASSES} classes."
            )
    except AttributeError:
        print("Could not verify NC of the aggregated model. Ensure model structure is as expected.")


    # 4. Save the full YOLO checkpoint (model, optimizer, etc.)
    # This makes it fully loadable by YOLO('path/to/pt') for the next round
    next_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num + 1}.pt')

    # The 'save' method of the YOLO object creates a proper checkpoint
    base_model_for_aggregation.save(next_global_model_path)
    print(f"✅ Aggregated global model for Round {round_num + 1} saved to: {next_global_model_path}")


def main():
    # Fix for PyTorch 2.6+ to avoid pickling errors (good to keep)
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

    # CRITICAL FIX: GLOBAL MODEL HEAD ADAPTATION
    # This ensures the global model's Detect head is configured for NUM_CLASSES (4)
    # before the first round of federated training.
    print(f"Server: Forcing global model head adaptation to {NUM_CLASSES} classes...")
    # Use any client's data.yaml for this adaptation
    DUMMY_DATA_YAML_FOR_ADAPTATION = CLIENTS[0]['data_yaml'] # Use client1's data.yaml for adaptation

    # Load initial yolov8n.pt model
    initial_global_yolo_model = YOLO('yolov8n.pt')

    # Run a single dummy epoch training to force head adaptation.
    # This creates a temporary 'runs' directory, which will be cleaned up.
    try:
        initial_global_yolo_model.train(
            data=DUMMY_DATA_YAML_FOR_ADAPTATION,
            epochs=1,  # Just 1 epoch to force adaptation
            imgsz=640,
            batch=1,   # Minimal batch size to reduce resource usage
            device='cuda', # Use cuda for adaptation
            project='_temp_global_adapter_project',  # Temporary project folder
            name='_global_head_adapter_run',         # Temporary run name
            exist_ok=True, # Allow overwriting temp folder if run multiple times
            save=False,    # Don't save weights from this dummy run
            plots=False,
            verbose=False, # Suppress verbose output
            workers=0 # No need for multiple workers for this quick dummy run
        )
    except Exception as e:
        print(f"WARNING: Dummy training for global model adaptation failed: {e}")
        print("This might be due to issues with dataset paths or ultralytics setup. Attempting to proceed.")

    # After this dummy training, the initial_global_yolo_model's head should be correctly adapted.
    # Save this adapted model as the initial global model for the first round.
    initial_global_yolo_model.save(INITIAL_GLOBAL_MODEL_PATH)
    print(f"✅ Initial global model (adapted to {NUM_CLASSES} classes) saved to: {INITIAL_GLOBAL_MODEL_PATH}")

    # Clean up the temporary training directory created by this dummy run.
    temp_adapter_dir = os.path.join('_temp_global_adapter_project', '_global_head_adapter_run')
    if os.path.exists(temp_adapter_dir):
        try:
            shutil.rmtree(temp_adapter_dir)
            print(f"Server: Cleaned up temporary directory: {temp_adapter_dir}")
        except Exception as e:
            print(f"Server: Could not remove temporary directory {temp_adapter_dir}: {e}")
    # --- END CRITICAL FIX ---


    for round_num in range(NUM_ROUNDS):
        print(f"\n===== Federated Learning Round {round_num + 1}/{NUM_ROUNDS} =====")
        weights_paths = train_clients(round_num) # This returns paths to client best.pt
        aggregate(weights_paths, round_num) # Pass round_num to aggregate for saving path
        print(f"✅ Completed Round {round_num + 1}\n")

if __name__ == "__main__":
    main()
