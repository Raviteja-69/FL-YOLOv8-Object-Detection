import os
from ultralytics import YOLO

Centralised_data_yaml = 'Datasets/construction_safetyV2_1206images/data.yaml'

model = YOLO('yolov8n.pt')

# FL was 3 rounds * 5 epochs/round = 15 total epochs
epochs = 15

OUTPUT_CENTRALIZED_DIR = 'centralized_training_results'
os.makedirs(OUTPUT_CENTRALIZED_DIR, exist_ok=True)

print(f"--- Starting Centralized Training for {epochs} epochs ---")

model.train(
    data=Centralised_data_yaml,
    epochs=epochs,
    imgsz=640,
    batch=16, # Adjust batch size based on your system's memory
    workers=4,
    device='cpu', # Or 'cuda:0' if you have a GPU
    project=OUTPUT_CENTRALIZED_DIR,
    name='centralized_yolov8n_cons_safety_data',
    plots=True
)
print(f"Centralized training complete. Model saved to {OUTPUT_CENTRALIZED_DIR}/centralized_yolov8n_cons_safety_data/weights/best.pt")
