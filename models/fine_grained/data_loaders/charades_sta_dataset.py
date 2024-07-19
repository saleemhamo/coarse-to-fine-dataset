# models/fine_grained/data_loaders/charades_sta_dataset.py
import os
import torch
from torch.utils.data import Dataset
from utils.logger import setup_logger

logger = setup_logger('charades_sta_dataset_logger')


class CharadesSTADatasetFineGrained(Dataset):
    def __init__(self, annotations, video_dir):
        self.annotations = annotations
        self.video_dir = video_dir
        logger.info(f"Initialized CharadesSTADataset with {len(annotations)} annotations.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_path = self.get_video_path(annotation['video_name'])
        video_frames = self.load_video_frames(video_path)
        text_sentence = annotation['sentence']
        label = torch.tensor(1)  # Assuming all pairs are positive examples for this task
        return video_frames, text_sentence, label

    def get_video_path(self, video_name):
        return os.path.join(self.video_dir, f"{video_name}.mp4")

    def load_video_frames(self, video_path):
        # Placeholder function to load video frames
        # Replace with actual video frame extraction logic
        frames = []
        # Example logic for loading video frames using OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
