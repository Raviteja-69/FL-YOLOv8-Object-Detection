import os
import subprocess
import torch
from ultralytics import YOLO

NUM_ROUNDS = 5
CLIENTS = [
    {'data_yaml': 'ppekit/client1/data.yaml', 'model_name': 'client1'},
    {'data_yaml': 'ppekit/client2/data.yaml', 'model_name': 'client2'},
]
GLOBAL_MODEL_SAVE_DIR = 'global_model_checkpoints'  # Directory to save checkpoints
os.makedirs(GLOBAL_MODEL_SAVE_DIR, exist_ok=True)  # Ensure directory exists
INITIAL_GLOBAL_MODEL_PATH = os.path.join(GLOBAL_MODEL_SAVE_DIR, 'global_model_r0.pt')

EPOCHS_PER_ROUND = 10
NUM_CLASSES = 4

def train_clients(round_num):
    print(f"\n=== Training Round {round_num + 1} ===")
    weights_paths = []

    # Define the current global model path
    current_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num}.pt')

    for client in CLIENTS:
        save_name = f"{client['model_name']}_round{round_num + 1}"

        # Determine starting weights for this round
        if round_num == 0:
            # For the very first round, use the original yolov8n.pt
            starting_weights = 'yolov8n.pt'
        else:
            # For subsequent rounds, use the *full checkpoint* of the aggregated global model
            # This ensures the model's architecture is loaded correctly.
            starting_weights = current_global_model_path

        print(f"  Client {client['model_name']} starting with weights: {starting_weights}")

        subprocess.run([
            'python', 'train_client.py',
            '--data', client['data_yaml'],
            '--save', save_name,  # This will create runs/detect/clientX_roundY/weights/best.pt
            '--epochs', str(EPOCHS_PER_ROUND),
            '--weights', starting_weights
        ], check=True)

        # Retrieve the path to the best.pt saved by ultralytics train command
        client_run_dir = os.path.join('runs', 'detect', save_name)
        best_pt_path = os.path.join(client_run_dir, 'weights', 'best.pt')

        if not os.path.exists(best_pt_path):
            raise FileNotFoundError(f"Client {client['model_name']} trained model not found at {best_pt_path}")

        weights_paths.append(best_pt_path)
    return weights_paths


def aggregate(weights_paths, round_num):
    import aggregate_weights  # Import your aggregation script

    # 1. Load the state_dicts from client models
    client_state_dicts = []
    for p in weights_paths:
        # Load the checkpoint as it is saved by ultralytics
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        # Extract the model's state_dict, which is usually under the 'model' key
        if 'model' in ckpt:
            if hasattr(ckpt['model'], 'state_dict'):
                client_state_dicts.append(ckpt['model'].state_dict())
            elif isinstance(ckpt['model'], dict):
                client_state_dicts.append(ckpt['model'])
            else:
                raise ValueError(f"Unexpected type for ckpt['model'] at {p}: {type(ckpt['model'])}")
        elif hasattr(ckpt, 'state_dict'):
            client_state_dicts.append(ckpt.state_dict())
        else:  # Assume ckpt itself is the state_dict
            client_state_dicts.append(ckpt)

    # 2. Average the state_dicts
    # We should use the modified average_weights that assumes consistent shapes
    avg_state_dict = aggregate_weights.average_weights(weights_paths)  # This will re-load inside
    # OR, modify average_weights to accept list of state_dicts:
    # avg_state_dict = average_weights_from_state_dicts(client_state_dicts)

    # 3. Create a new global model with the correct structure (nc=4)
    # For the first round, the client models will have adapted their heads to 4 classes.
    # The aggregated model should also have 4 classes.
    # So, we load the initial yolov8n.pt, then load the aggregated state_dict.
    # The best way is to load one of the trained client models
    # (which has the correct 4-class structure) and then update its weights.

    # Load one of the client models to get its structure (which should be 4-class)
    # Assuming all client models are trained to 4 classes, pick any one.
    base_model_for_aggregation = YOLO(weights_paths[0])

    # Apply the averaged state_dict to this base model
    base_model_for_aggregation.model.load_state_dict(avg_state_dict)

    # Now, verify the nc of the aggregated model
    print(f"Aggregated Model NC: {base_model_for_aggregation.model.model[-1].nc}")
    if base_model_for_aggregation.model.model[-1].nc != NUM_CLASSES:
        print(
            f"WARNING: Aggregated model nc is {base_model_for_aggregation.model.model[-1].nc}, expected {NUM_CLASSES}")

    # 4. Save the full YOLO checkpoint (model, optimizer, etc.)
    # This makes it fully loadable by YOLO('path/to/pt') for the next round
    next_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num + 1}.pt')

    # The 'save' method of the YOLO object creates a proper checkpoint
    base_model_for_aggregation.save(next_global_model_path)
    print(f"✅ Aggregated global model for Round {round_num + 1} saved to: {next_global_model_path}")


def main():
    # Fix for PyTorch 2.6+ to avoid pickling errors (good to keep)
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

    # Initialize the first global model (yolov8n.pt) if it doesn't exist in our save directory
    # This ensures global_model_r0.pt exists for Round 1 client training
    if not os.path.exists(INITIAL_GLOBAL_MODEL_PATH):
        print(f"Creating initial global model checkpoint at {INITIAL_GLOBAL_MODEL_PATH}")
        YOLO('yolov8n.pt').save(INITIAL_GLOBAL_MODEL_PATH)
    else:
        print(f"Initial global model checkpoint found at {INITIAL_GLOBAL_MODEL_PATH}")

    for round_num in range(NUM_ROUNDS):
        weights_paths = train_clients(round_num)  # This returns paths to client best.pt
        aggregate(weights_paths, round_num)  # Pass round_num to aggregate for saving path
        print(f"✅ Completed Round {round_num + 1}\n")

if __name__ == "__main__":
    main()
