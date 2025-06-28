import torch
from copy import deepcopy

def federated_average(state_dicts):
    """
    Performs federated averaging on a list of model state dictionaries.

    Args:
        state_dicts (list[dict]): A list of model state_dicts to be averaged.

    Returns:
        dict: A new state_dict with the averaged weights. Returns an empty dict if input is empty.
    """
    if not state_dicts:
        return {}

    # Use the first state_dict as a template for the aggregated weights
    avg_state_dict = deepcopy(state_dicts[0])

    # Zero out the template to ensure a clean slate
    for key in avg_state_dict:
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

    # Sum all the state_dicts
    for sd in state_dicts:
        for key in avg_state_dict:
            if key in sd:
                # Add a shape check for robustness
                if avg_state_dict[key].shape == sd[key].shape:
                    avg_state_dict[key] += sd[key]
                else:
                    print(f"Warning: Skipping key '{key}' due to shape mismatch during aggregation.")
            else:
                print(f"Warning: Skipping key '{key}' as it is missing in one of the client models.")

    # Divide by the number of clients to get the average
    num_clients = len(state_dicts)
    if num_clients > 0:
        for key in avg_state_dict:
            avg_state_dict[key] = torch.div(avg_state_dict[key], num_clients)

    return avg_state_dict 
