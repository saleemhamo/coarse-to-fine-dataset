from data.charades_sta import CharadesSTA
from utils.constants import *


def get_dataset(dataset_name):
    """
    Function to get the appropriate dataset handler based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        dataset_handler: Instance of the dataset handler.
    """
    if dataset_name == CHARADES_STA:
        return CharadesSTA(
            video_dir=CHARADES_VIDEOS_DIR,
            train_file=CHARADES_ANNOTATIONS_TRAIN,
            test_file=CHARADES_ANNOTATIONS_TEST
        )
    else:
        raise ValueError("Unsupported dataset name")
