import os
from datetime import datetime

import torch


def save_model(model, directory, model_name=None, key=None, custom_file_name=None):
    max_file_name_length = 255  # Maximum file name length in Linux

    if custom_file_name:
        filename = custom_file_name
    else:
        if key is None:
            key = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{key}.pth"

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

    # Ensure the file name does not exceed the max length
    if len(filename) > max_file_name_length:
        filename = filename[:max_file_name_length]

    path = os.path.join(directory, filename)
    print(f"Saving model to: {path}")

    # Check directory permissions
    dir_permissions = os.stat(directory)
    print(f"Directory permissions: {dir_permissions}")

    # Attempt to open the file to check for errors
    try:
        with open(path, 'wb') as f:
            pass
    except Exception as e:
        print(f"Error opening file for writing: {e}")

    torch.save(model.state_dict(), path)
    return path


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def get_device(logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if logger:
        logger.info(f"Using device: {device}")
    else:
        print(f"Using device: {device}")
    return device
