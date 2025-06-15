import torch
from ultralytics.nn.modules import Detect # We might need this for proper model reconstruction

def average_weights(weight_paths):
    """
    Load model weights from given paths and average their state_dicts.
    Assumes all models are trained on the same number of classes (nc)
    and thus have compatible shapes for their detection heads.
    """
    state_dicts = []
    for p in weight_paths:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        # YOLOv8 checkpoints can be a bit tricky, 'model' key usually holds the actual model or its state_dict
        if 'model' in ckpt:
            # If ckpt['model'] is a nn.Module, get its state_dict
            if hasattr(ckpt['model'], 'state_dict'):
                state_dict = ckpt['model'].state_dict()
            # If ckpt['model'] is already a state_dict
            elif isinstance(ckpt['model'], dict):
                state_dict = ckpt['model']
            else:
                raise ValueError(f"Unexpected type for ckpt['model']: {type(ckpt['model'])}")
        elif hasattr(ckpt, 'state_dict'): # For cases where ckpt itself is a model
            state_dict = ckpt.state_dict()
        else: # If ckpt is already the state_dict
            state_dict = ckpt
        state_dicts.append(state_dict)

    if not state_dicts:
        raise ValueError("No model weights loaded for averaging.")

    # All client models should have the same structure after their first training round
    # If the first client trained on 4 classes, its head should be 4 classes.
    # We'll use the first state_dict as the reference for structure.
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        # Ensure the key exists in all other state_dicts and shapes match
        if all(key in sd and sd[key].shape == state_dicts[0][key].shape for sd in state_dicts[1:]):
            avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
        else:
            # This indicates an inconsistency, which shouldn't happen if clients are consistent
            # and trained on the same data configuration.
            print(f"Warning: Skipping key '{key}' due to shape or key mismatch across client models. "
                  "This might indicate an issue with client model consistency.")
            # For debugging, you might want to print the shapes for the problematic key
            # print([sd[key].shape for sd in state_dicts if key in sd])

    return avg_state_dict


def save_averaged_model(avg_state_dict, save_path):
    """
    Save averaged weights in a format compatible with YOLOv8.
    This means saving it as a dictionary under the 'model' key,
    along with other necessary info that a YOLOv8 checkpoint usually has.
    """
    # A standard YOLOv8 checkpoint has keys like 'model', 'ema', 'optimizer', 'epoch', etc.
    # To make it fully loadable by YOLO('path/to/model.pt'), we need to mimic this.
    # For simple aggregation, we often just need 'model'.
    # For full compatibility, you might copy other parts from a 'best.pt' file.

    # Option 1: Minimal saving (may require manual nc update when loading in main.py)
    # torch.save({'model': avg_state_dict}, save_path)

    # Option 2: More complete saving (better for direct YOLO() loading)
    # Get a sample checkpoint to extract other metadata like 'nc', 'names', 'args', 'ema' etc.
    # This assumes all client models originated from the same base and have similar metadata.
    # A better approach is to create a *dummy* YOLO model with the correct structure (nc=4)
    # and then load these weights into its state_dict. This ensures `nc` is correctly set.

    # Let's refine this in main.py, as saving just the state_dict here is simpler.
    # The crucial part is loading this `avg_state_dict` into a correctly structured YOLOv8 model.
    torch.save(avg_state_dict, save_path) # Save just the state_dict for now
    print(f"✅ Averaged state_dict saved to: {save_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', required=True, help="List of model weights to average")
    parser.add_argument('--save', type=str, required=True, help="Path to save averaged model")
    args = parser.parse_args()

    avg_state_dict = average_weights(args.weights)
    save_averaged_model(avg_state_dict, args.save)
