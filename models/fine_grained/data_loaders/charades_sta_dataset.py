# models/fine_grained/data_loaders/charades_sta_dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.logger import setup_logger

logger = setup_logger('charades_sta_dataset_logger')


class CharadesSTADatasetFineGrained(Dataset):
    def __init__(self, annotations, video_dir, target_size=(224, 224)):
        self.annotations = annotations
        self.video_dir = video_dir
        self.target_size = target_size
        logger.info(f"Initialized CharadesSTADataset with {len(annotations)} annotations.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_features = self.load_and_preprocess_video(self.get_video_path(annotation['video_name']))
        text_features = annotation['sentence']
        label = torch.tensor(1)  # Assuming all pairs are positive examples for this task
        return video_features, text_features, label

    def get_video_path(self, video_name):
        return os.path.join(self.video_dir, f"{video_name}.mp4")

    def load_and_preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        cap.release()
        return np.array(frames)
