import os
from ultralytics import YOLO

def client_train_task(args):
    """
    The main task for a client in a federated learning round.
    Loads the global model, trains it on local data, and returns the trained model's state_dict.
    """
    client_id, global_model_path, data_yaml, epochs, output_dir, current_round, device = args

    print(f"  Client {client_id}: Starting local training (Round {current_round + 1})...")

    # Clients load the global model as a .pt file
    client_model = YOLO(global_model_path)

    # Define a unique name for this client's training run
    client_run_name = f'{client_id}_round_{current_round}'
    client_output_path = os.path.join(output_dir, client_run_name)

    # Train the model
    client_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        project=output_dir,
        name=client_run_name,
        verbose=False, # Suppress verbose ultralytics output for cleaner logs
        save=False, # We only need the final model state, not intermediate checkpoints
        save_period=-1,
        plots=False
    )

    # The trained model is in the 'model' attribute of the YOLO object.
    # We return its state_dict for aggregation.
    best_client_model = client_model.model
    print(f"  Client {client_id}: Local training finished. Model state_dict prepared.")

    # Save the trained model as a full checkpoint
    best_model_path = os.path.join(client_output_path, 'weights', 'best.pt')
    client_model.save(best_model_path)
    print(f"  Client {client_id}: Local training finished. Model saved at {best_model_path}")
    return best_model_path 