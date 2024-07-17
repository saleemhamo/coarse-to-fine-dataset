import os
from datetime import datetime

import torch


def save_model(model, directory, model_name=None, key=None, custom_file_name=None):
    if custom_file_name:
        filename = custom_file_name
    else:
        if key is None:
            key = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{key}.pth"

    # date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{filename}_{date}.pth"

    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

    path = os.path.join(directory, filename)
    print(f"Saving model to: {path}")
    torch.save(model.state_dict(), path)
    return path


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
