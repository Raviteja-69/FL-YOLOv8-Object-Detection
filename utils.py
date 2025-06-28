import os
import shutil
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def adapt_and_save_initial_model(num_classes, data_yaml_path, save_path, device='cpu'):
    """
    Adapts a pre-trained YOLOv8n model head to the desired number of classes using dummy training,
    saves the adapted model, and cleans up temporary files.

    Args:
        num_classes (int): The target number of classes.
        data_yaml_path (str): Path to the data.yaml file to use for adaptation.
        save_path (str): The file path to save the adapted initial global model.
        device (str): The device to use for dummy training ('cpu' or 'cuda').
    """
    logger.info(f"Adapting initial global model head to {num_classes} classes...")

    # Prevent re-adaptation if the model already exists
    if os.path.exists(save_path):
        logger.info(f"Initial model at {save_path} already exists. Skipping adaptation.")
        return

    initial_global_yolo_model = YOLO('yolov8n.pt')
    temp_project_dir = '_temp_global_adapter_project'
    temp_run_name = '_global_head_adapter_run'
    temp_full_dir = os.path.join(temp_project_dir, temp_run_name)

    try:
        initial_global_yolo_model.train(
            data=data_yaml_path,
            epochs=1,
            imgsz=640,
            batch=1,
            device=device,
            project=temp_project_dir,
            name=temp_run_name,
            exist_ok=True,
            save=False,
            plots=False,
            verbose=False,
            workers=0
        )
        logger.info("Dummy training for head adaptation completed successfully.")
    except Exception as e:
        logger.error(f"Dummy training for global model adaptation FAILED: {e}")
        logger.error("Please check your dataset paths in config.yaml and ensure ultralytics is set up correctly.")
        # Re-raise the exception to halt execution, as this is a critical step
        raise e

    try:
        # Ensure the directory for the save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        initial_global_yolo_model.save(save_path)
        logger.info(f"✅ Initial global model (adapted to {num_classes} classes) saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save initial adapted global model: {e}")
        raise e

    # Clean up the temporary directory
    if os.path.exists(temp_project_dir):
        try:
            shutil.rmtree(temp_project_dir)
            logger.info(f"Cleaned up temporary adaptation directory: {temp_project_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_project_dir}: {e}") 