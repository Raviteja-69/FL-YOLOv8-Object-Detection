import os
import shutil
import logging
import torch
from ultralytics import YOLO
import tempfile
import yaml

logger = logging.getLogger(__name__)

def adapt_and_save_initial_model(num_classes, data_yaml_path, save_path, device='cpu'):
    """
    Adapts a pre-trained YOLOv8n model head to the desired number of classes using zero-shot adaptation.
    This method modifies the classification layer without any training, eliminating bias.

    Args:
        num_classes (int): The target number of classes.
        data_yaml_path (str): Path to the data.yaml file (used only for validation, not training).
        save_path (str): The file path to save the adapted initial global model.
        device (str): The device to use for model operations ('cpu' or 'cuda').
    """
    logger.info(f"Adapting initial global model head to {num_classes} classes using zero-shot adaptation...")

    # Prevent re-adaptation if the model already exists
    if os.path.exists(save_path):
        logger.info(f"Initial model at {save_path} already exists. Skipping adaptation.")
        return

    try:
        # Load the pre-trained model
        model = YOLO('yolov8n.pt')
        yolo_model = model.model
        if yolo_model is None or not hasattr(yolo_model, 'named_modules'):
            raise RuntimeError("YOLO model is not loaded correctly or does not have named_modules.")

        # Find the Detect head (usually last module with 'nc' attribute and 'cv3')
        detect_head = None
        for name, module in yolo_model.named_modules():
            if hasattr(module, 'nc') and hasattr(module, 'cv3'):
                detect_head = module
                break
        if detect_head is None:
            raise RuntimeError("Could not find Detect head in YOLOv8 model.")

        old_nc = detect_head.nc
        detect_head.nc = num_classes
        logger.info(f"Modified Detect layer: {old_nc} classes -> {num_classes} classes")

        # For each detection scale, replace the final Conv2d layer
        for i, seq in enumerate(detect_head.cv3):
            if hasattr(seq, '__getitem__') and len(seq) > 2:
                last_conv = seq[2]
                if isinstance(last_conv, torch.nn.Conv2d):
                    in_channels = last_conv.in_channels
                    out_channels = num_classes
                    # Force kernel_size, stride, and padding to be tuples of exactly two ints
                    def to_tuple2(val, default):
                        if isinstance(val, int):
                            return (val, val)
                        elif isinstance(val, tuple):
                            if len(val) == 2:
                                return (int(val[0]), int(val[1]))
                            elif len(val) > 2:
                                return (int(val[0]), int(val[1]))
                            elif len(val) == 1:
                                return (int(val[0]), int(val[0]))
                        return default
                    kernel_size = to_tuple2(last_conv.kernel_size, (1, 1))
                    stride = to_tuple2(last_conv.stride, (1, 1))
                    padding = to_tuple2(last_conv.padding, (0, 0))
                    bias = last_conv.bias is not None
                    # Create new Conv2d layer
                    new_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                    torch.nn.init.normal_(new_conv.weight, mean=0.0, std=0.01)
                    if bias and new_conv.bias is not None:
                        torch.nn.init.constant_(new_conv.bias, 0.0)
                    seq[2] = new_conv
                    logger.info(f"Replaced cv3[{i}][2] Conv2d: {last_conv.out_channels} -> {num_classes} classes")
                else:
                    logger.warning(f"cv3[{i}][2] is not Conv2d, skipping.")
            else:
                logger.warning(f"cv3[{i}] is not a Sequential with at least 3 layers, skipping.")

        # Update the underlying model's class names and nc attribute
        yolo_model.names = {i: f'class_{i}' for i in range(num_classes)}
        yolo_model.nc = num_classes

        # Ensure the directory for the save_path exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save the adapted model
        model.save(save_path)
        logger.info(f"✅ Initial global model (adapted to {num_classes} classes) saved to: {save_path}")
        logger.info("Zero-shot head adaptation completed successfully - no training bias introduced.")

    except Exception as e:
        logger.error(f"Zero-shot head adaptation FAILED: {e}")
        logger.error("Please check your model and ensure ultralytics is set up correctly.")
        # Re-raise the exception to halt execution, as this is a critical step
        raise e
