import os
from ultralytics import YOLO

def train_client(data_path, model_save_path, epochs=10, weights_path='yolov8n.pt'):
    """
    Train YOLOv8 model on a client's local data.

    Args:
        data_path (str): Path to client dataset YAML file.
        model_save_path (str): Name to save the trained model weights.
        epochs (int): Number of training epochs.
        weights_path (str): Path to starting weights file (default: yolov8n.pt).
    """
    model = YOLO(weights_path)  # Load specified weights!

    # Train on client's data
    model.train(data=data_path, epochs=epochs, imgsz=640, name=model_save_path, save=True)

    print(f"Training completed for client. Model saved at {model_save_path}.pt")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to client's data YAML file")
    parser.add_argument('--save', type=str, required=True, help="Name to save the trained model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help="Starting weights path (default: yolov8n.pt)")
    args = parser.parse_args()

    train_client(args.data, args.save, args.epochs, args.weights)
