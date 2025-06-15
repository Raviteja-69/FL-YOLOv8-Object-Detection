import os
from ultralytics import YOLO

Centralised_data_yaml = 'Data/centralized_data.yaml'

model = YOLO('yolov8n.pt')

epochs = 15

OUTPUT_CENTRALIZED_DIR = 'centralized_training_results'
os.makedirs(OUTPUT_CENTRALIZED_DIR, exist_ok=True)

print(f"--- Starting Centralized Training for {epochs} epochs ---")

model.train(
    data=Centralised_data_yaml,
    epochs=epochs,
    imgsz=640,
    batch=16, # Adjust batch size based on your system's memory
    device='cpu', # Or 'cuda:0' if you have a GPU
    project=OUTPUT_CENTRALIZED_DIR,
    name='centralized_yolov8n_ppe_dataset',
    plots=True
)
print(f"Centralized training complete. Model saved to {OUTPUT_CENTRALIZED_DIR}/centralized_yolov8n_ppe_dataset/weights/best.pt")
