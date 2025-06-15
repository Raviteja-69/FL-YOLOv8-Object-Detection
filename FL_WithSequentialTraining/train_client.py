import os
from ultralytics import YOLO
import argparse # Import argparse

def train_client(data_path, model_save_name, epochs=10, weights_path='yolov8n.pt', device='cpu', workers=8):
    """
    Train YOLOv8 model on a client's local data.

    Args:
        data_path (str): Path to client dataset YAML file.
        model_save_name (str): Name to save the trained model weights (ultralytics uses this for the run directory).
        epochs (int): Number of training epochs.
        weights_path (str): Path to starting weights file (default: yolov8n.pt).
        device (str): Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0').
        workers (int): Number of workers for data loading.
    """
    model = YOLO(weights_path)  # Load specified weights!

    # Train on client's data
    # 'name' parameter controls the directory name within 'runs/detect'
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        name=model_save_name, 
        save=True, # Saves best.pt and last.pt
        device=device, # Use the passed device
        workers=workers, # Use the passed number of workers
        verbose=False, # Suppress excessive output if running many clients
        # Disable plots and save_txt/save_conf for cleaner logs and less temporary file generation
        plots=False,
        save_txt=False,
        save_conf=False
    )

    print(f"Training completed for client. Model results saved under runs/detect/{model_save_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to client's data YAML file")
    parser.add_argument('--save_name', type=str, required=True, help="Name for the training run directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help="Starting weights path (default: yolov8n.pt)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training (e.g., 'cpu', 'cuda')")
    parser.add_argument('--workers', type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    train_client(args.data, args.save_name, args.epochs, args.weights, args.device, args.workers)
