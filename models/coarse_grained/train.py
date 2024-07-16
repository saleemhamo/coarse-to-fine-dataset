import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from data.charades_sta import CharadesSTA
from models.coarse_grained.model import CoarseGrainedModel
import torch.optim as optim
import torch.nn as nn
from utils.config import Config
from utils.model_utils import save_model, get_device
from utils.constants import CHARADES_VIDEOS_DIR, CHARADES_ANNOTATIONS_TRAIN, CHARADES_ANNOTATIONS_TEST, \
    COARSE_GRAINED_MODELS_DIR
from utils.logger import setup_logger
import os
import cv2
import numpy as np

# Setup logger
logger = setup_logger('train_logger')


class CharadesSTADataset(Dataset):
    def __init__(self, annotations, video_dir, clip_model, clip_processor):
        self.annotations = annotations
        self.video_dir = video_dir
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        logger.info(f"Initialized CharadesSTADataset with {len(annotations)} annotations.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_features = self.extract_video_features(annotation['video_name'])
        text_features = self.extract_text_features(annotation['sentence'])
        label = torch.tensor(1)  # Assuming all pairs are positive examples for this task
        return video_features, text_features, label

    def extract_video_features(self, video_name):
        frames = self.load_video(video_name)
        video_features = []
        for frame in frames:
            inputs = self.clip_processor(images=frame, return_tensors="pt")
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs).squeeze(0)
            video_features.append(features)
        logger.info(f"Extracted video features for {video_name}")
        return torch.stack(video_features).mean(dim=0)  # Averaging features of all frames

    def extract_text_features(self, sentence):
        inputs = self.clip_processor(text=sentence, return_tensors="pt")
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs).squeeze(0)
        logger.info(f"Extracted text features for sentence: {sentence}")
        return text_features

    def load_video(self, video_name):
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        logger.info(f"Loading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from video {video_name}")
        return np.array(frames)


def train_coarse_grained_model(train_loader, config):
    device = get_device()
    video_dim = train_loader.dataset[0][0].size(-1)
    text_dim = train_loader.dataset[0][1].size(-1)
    model = CoarseGrainedModel(video_dim, text_dim, hidden_dim=512).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.coarse_grained['learning_rate'])

    logger.info(f"Starting training for {config.coarse_grained['num_epochs']} epochs.")
    for epoch in range(config.coarse_grained['num_epochs']):
        model.train()
        total_loss = 0
        for video_features, text_features, labels in train_loader:
            video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(
                device)
            optimizer.zero_grad()
            outputs = model(video_features, text_features)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}/{config.coarse_grained['num_epochs']}, Loss: {total_loss / len(train_loader)}")

    logger.info("Training completed.")
    return model


def main():
    logger.info("Loading configuration.")
    config = Config()
    charades_sta = CharadesSTA(
        video_dir=CHARADES_VIDEOS_DIR,
        train_file=CHARADES_ANNOTATIONS_TRAIN,
        test_file=CHARADES_ANNOTATIONS_TEST
    )
    annotations = charades_sta.get_train_data()

    # Load CLIP model and processor
    logger.info("Loading CLIP model and processor.")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = CharadesSTADataset(annotations, CHARADES_VIDEOS_DIR, clip_model, clip_processor)
    train_loader = DataLoader(dataset, batch_size=config.coarse_grained['batch_size'], shuffle=True)
    logger.info("Data loader created.")

    model = train_coarse_grained_model(train_loader, config)

    docker_image_name = os.getenv('DOCKER_IMAGE_NAME', 'default_image')
    job_id = os.getenv('JOB_ID', 'default_job')
    model_file_name = f"model_{docker_image_name}_{job_id}.pth"

    save_path = save_model(model, COARSE_GRAINED_MODELS_DIR, custom_file_name=model_file_name)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
