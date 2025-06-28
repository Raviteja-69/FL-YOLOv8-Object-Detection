import os
import sys
import yaml
import logging
from multiprocessing import Pool
from ultralytics import YOLO


# --- Add project root to sys.path ---
# This allows imports from core to work regardless of how the script is run.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our new modular components
from FL_WithMultiprocessing.client import client_train_task
from core.aggregation import federated_average
from core.utils import adapt_and_save_initial_model


# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('federated_learning_mp.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG LOADING ===
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# --- Server Configuration ---
NUM_ROUNDS = config['num_rounds']
EPOCHS_PER_ROUND = config['epochs_per_round']
NUM_CLIENTS_PER_ROUND = config['num_clients_per_round']
NUM_CLASSES = config['num_classes']
DEVICE = config['device']
CLIENTS = config['clients']

# --- Directory Setup ---
FL_PROJECT_OUTPUTS_DIR = config['fl_project_outputs_dir']
CLIENT_MODEL_DIR = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'client_models')
GLOBAL_MODEL_CHECKPOINTS_DIR = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'global_model_checkpoints')
os.makedirs(CLIENT_MODEL_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_CHECKPOINTS_DIR, exist_ok=True)

def main():
    """
    Main server loop for multiprocessing-based federated learning.
    """
    # --- Initial Model Adaptation ---
    initial_global_model_path = os.path.join(GLOBAL_MODEL_CHECKPOINTS_DIR, 'global_model_r0.pt')
    logger.info(f"Adapting initial global model head to {NUM_CLASSES} classes (if needed)...")
    adapt_and_save_initial_model(
        num_classes=NUM_CLASSES,
        data_yaml_path=CLIENTS[0]['data_yaml'],
        save_path=initial_global_model_path,
        device=DEVICE
    )
    logger.info(f"Initial global model for FL saved at {initial_global_model_path}")

    # --- Federated Learning Server Loop ---
    # Load the adapted model to start the FL process
    global_model = YOLO(initial_global_model_path)

    for current_round in range(NUM_ROUNDS):
        logger.info(f"\n===== Federated Learning Round {current_round + 1}/{NUM_ROUNDS} =====")

        # Save the current global model for clients to load
        current_global_model_path = os.path.join(GLOBAL_MODEL_CHECKPOINTS_DIR, f'global_model_r{current_round}.pt')
        global_model.save(current_global_model_path)
        logger.info(f"Server: Global model for round {current_round + 1} saved at {current_global_model_path}")

        # --- Client Selection & Parallel Training ---
        # For simplicity, we select all clients. In a real scenario, you might select a subset.
        selected_clients = CLIENTS[:NUM_CLIENTS_PER_ROUND]
        client_tasks = []
        for client in selected_clients:
            client_tasks.append(
                (client['model_name'], current_global_model_path, client['data_yaml'], EPOCHS_PER_ROUND, CLIENT_MODEL_DIR, current_round, DEVICE)
            )

        client_state_dicts = []
        # Use a process pool to train clients in parallel
        with Pool(processes=os.cpu_count()) as pool:
            logger.info(f"Server: Starting parallel training for {len(client_tasks)} clients...")
            results = pool.map(client_train_task, client_tasks)
            for path in results:
                model = YOLO(path)
                client_state_dicts.append(model.model.state_dict())

        # --- Aggregation ---
        if client_state_dicts:
            logger.info(f"Server: Aggregating models from {len(client_state_dicts)} clients...")
            # Perform federated averaging
            avg_state_dict = federated_average(client_state_dicts)
            # Load the averaged weights into the global model
            if hasattr(global_model, 'model') and global_model.model is not None:
                global_model.model.load_state_dict(avg_state_dict)
                logger.info("Server: Aggregation complete.")
            else:
                logger.error("Could not load state_dict, global_model.model is not a valid module.")
        else:
            logger.warning("Server: No client models received for aggregation. Skipping round.")

    # --- Save Final Model ---
    final_global_model_path = os.path.join(GLOBAL_MODEL_CHECKPOINTS_DIR, 'global_model_final.pt')
    global_model.save(final_global_model_path)
    logger.info(f"\nFederated Learning training finished. Final model saved to {final_global_model_path}")

if __name__ == "__main__":
    main() 