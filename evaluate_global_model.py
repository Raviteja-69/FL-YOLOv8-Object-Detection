import os
from ultralytics import YOLO

VALIDATION_DATA_YAML = 'validation_data.yaml'

# List of global models to evaluate
GLOBAL_MODELS = [
    'global_model_checkpoints/global_model_r0.pt',  # The initial model
    'global_model_checkpoints/global_model_r1.pt',
    'global_model_checkpoints/global_model_r2.pt',
    'global_model_checkpoints/global_model_r3.pt',
    'global_model_checkpoints/global_model_r4.pt',
    'global_model_checkpoints/global_model_r5.pt'
]

def evaluate_model(model_path, validation_data_yaml):
    """
    Evaluates a YOLOv8 model and prints its performance metrics.
    """
    print(f"\n--- Evaluating Model: {model_path} ---")
    model = YOLO(model_path)

    # The 'val' method returns a Metrics object containing the results.
    # We can specify other arguments like imgsz, batch size, device.
    results = model.val(
        data=validation_data_yaml,
        imgsz=640,
        batch=16,
        device='cpu',  # Or 'cuda:0' if you have a GPU
        split='val',
        conf=0,
        iou=0,
        plots=True,
        project='evaluation_results',  #Project dir for saving results
        name=os.path.basename(model_path).replace('.pt', '')  # Name of the run for unique folder
    )
    print(f"  mAP50-95: {results.box.map}")
    print(f"  mAP50:    {results.box.map50}")
    print(f"  Precision: {results.box.p}")
    print(f"  Recall:    {results.box.r}")
    print(f"  F1 Score:  {results.box.f1}")

if __name__ == "__main__":
    for model_file in GLOBAL_MODELS:
        evaluate_model(model_file, VALIDATION_DATA_YAML)