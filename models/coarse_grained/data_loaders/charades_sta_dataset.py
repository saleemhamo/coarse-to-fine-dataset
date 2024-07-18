import os
import torch
from torch.utils.data import Dataset
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('charades_sta_dataset_logger')


class CharadesSTADataset(Dataset):
    def __init__(self, annotations, video_dir, feature_extractor):
        self.annotations = annotations
        self.video_dir = video_dir
        self.feature_extractor = feature_extractor
        logger.info(f"Initialized CharadesSTADataset with {len(annotations)} annotations.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_features = self.feature_extractor.extract_video_features(self.get_video_path(annotation['video_name']))
        text_features = self.feature_extractor.extract_text_features(annotation['sentence'])
        label = torch.tensor(1)  # Assuming all pairs are positive examples for this task
        return video_features, text_features, label

    def get_video_path(self, video_name):
        return os.path.join(self.video_dir, f"{video_name}.mp4")
