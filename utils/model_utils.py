import torch
import os
from datetime import datetime


def save_model(model, model_name, directory, key=None):
    if key is None:
        key = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{key}.pth"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    return path


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
