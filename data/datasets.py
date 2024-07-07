import os
import json
import cv2
import numpy as np
from data.charades_sta import CharadesSTA
from utils.constants import CHARADES_VIDEO_DIR, CHARADES_ANNOTATION_FILE


def get_dataset(dataset_name, video_dir=None, annotation_file=None):
    """
    Function to get the appropriate dataset handler based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        video_dir (str): Path to the directory containing video files.
        annotation_file (str): Path to the annotation file.

    Returns:
        dataset_handler: Instance of the dataset handler.
    """
    if dataset_name == "charades_sta":
        return CharadesSTA(video_dir=video_dir or CHARADES_VIDEO_DIR,
                           annotation_file=annotation_file or CHARADES_ANNOTATION_FILE)
    else:
        raise ValueError("Unsupported dataset name")


class GenericDataset:
    def __init__(self, video_dir, annotation_file):
        self.video_dir = video_dir
        self.annotation_file = annotation_file
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            return json.load(f)

    def get_video_path(self, video_id):
        return os.path.join(self.video_dir, f"{video_id}.mp4")

    def load_video(self, video_id):
        video_path = self.get_video_path(video_id)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)
