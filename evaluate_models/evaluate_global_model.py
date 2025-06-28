"""
evaluate_global_model.py

This script evaluates all YOLOv8 global model checkpoints in a directory on a validation dataset.
It prints key performance metrics for each model and class to the console for easy manual recording or further processing.

Usage:
    python evaluate_global_model.py --models_dir <dir> --val_yaml <path> --device <cpu/cuda:0> --output_dir <output_dir>
    (defaults: models_dir='global_model_checkpoints', val_yaml='validation_data.yaml', device='cpu', output_dir='evaluation_results')
"""

import os
import argparse
import glob
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all YOLOv8 global models in a directory.")
    parser.add_argument('--models_dir', type=str, default='global_model_checkpoints', help='Directory with global model .pt files')
    parser.add_argument('--val_yaml', type=str, default='validation_data.yaml', help='Validation data YAML file')
    parser.add_argument('--device', type=str, default='cpu', help="Device to use: 'cpu' or 'cuda:0'")
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save YOLO evaluation plots/results')
    return parser.parse_args()


def evaluate_model(model_path, validation_data_yaml, device, output_dir):
    print(f"\n--- Evaluating Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    try:
        model = YOLO(model_path)
        class_names = model.names
        nc = len(class_names)
        print(f"  Number of classes: {nc}")
        print(f"  Class names: {list(class_names.values())}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return
    try:
        results = model.val(
            data=validation_data_yaml,
            imgsz=640,
            batch=16,
            device=device,
            split='val',
            conf=0,
            iou=0,
            plots=True,
            project=output_dir,
            name=os.path.basename(model_path).replace('.pt', '')
        )
        # Print overall metrics
        print(f"  mAP50-95: {results.box.map}")
        print(f"  mAP50:    {results.box.map50}")
        # Print mean precision, recall, F1 if available
        if hasattr(results.box.p, 'mean'):
            print(f"  Precision (mean): {results.box.p.mean()}")
        else:
            print(f"  Precision: {results.box.p}")
        if hasattr(results.box.r, 'mean'):
            print(f"  Recall (mean):    {results.box.r.mean()}")
        else:
            print(f"  Recall:    {results.box.r}")
        if hasattr(results.box.f1, 'mean'):
            print(f"  F1 (mean):        {results.box.f1.mean()}")
        else:
            print(f"  F1:        {results.box.f1}")
        # Print class-wise metrics
        print("\n  Class-wise metrics:")
        print(f"    {'Class':<15}{'AP50-95':>10}{'Precision':>12}{'Recall':>12}{'F1':>12}")
        for cid, cname in model.names.items():
            ap = results.box.maps[cid] if hasattr(results.box, 'maps') else ''
            p = results.box.p[cid] if hasattr(results.box, 'p') else ''
            r = results.box.r[cid] if hasattr(results.box, 'r') else ''
            f1 = results.box.f1[cid] if hasattr(results.box, 'f1') else ''
            print(f"    {cname:<15}{ap:>10.3f}{p:>12.3f}{r:>12.3f}{f1:>12.3f}")
    except Exception as e:
        print(f"Error during evaluation of {model_path}: {e}")
        return


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_files = sorted(glob.glob(os.path.join(args.models_dir, '*.pt')))
    if not model_files:
        print(f"No model checkpoints found in {args.models_dir}")
        return
    for model_file in model_files:
        evaluate_model(model_file, args.val_yaml, args.device, args.output_dir)

if __name__ == "__main__":
    main()
    
