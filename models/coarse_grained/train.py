import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

import data
from models.coarse_grained.model import CoarseGrainedModel
import torch.optim as optim
import torch.nn as nn
from utils.config import Config
from utils.model_utils import save_model, get_device
from utils.constants import *
import os
import cv2
import numpy as np


class CharadesSTADataset(Dataset):
    def __init__(self, annotations, video_dir, clip_model, clip_processor):
        self.annotations = annotations
        self.video_dir = video_dir
        self.clip_model = clip_model
        self.clip_processor = clip_processor

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
        return torch.stack(video_features).mean(dim=0)  # Averaging features of all frames

    def extract_text_features(self, sentence):
        inputs = self.clip_processor(text=sentence, return_tensors="pt")
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs).squeeze(0)
        return text_features

    def load_video(self, video_name):
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.array(frames)


def train_coarse_grained_model(train_loader, config):
    device = get_device()
    video_dim = train_loader.dataset[0][0].size(-1)
    text_dim = train_loader.dataset[0][1].size(-1)
    model = CoarseGrainedModel(video_dim, text_dim, hidden_dim=512).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.coarse_grained['learning_rate'])

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
        print(f"Epoch {epoch + 1}/{config.coarse_grained['num_epochs']}, Loss: {total_loss / len(train_loader)}")

    return model


def main():
    config = Config()
    charades_sta = data.get_dataset(CHARADES_STA)
    annotations = charades_sta.get_train_data()

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = CharadesSTADataset(annotations, CHARADES_VIDEOS_DIR, clip_model, clip_processor)
    train_loader = DataLoader(dataset, batch_size=config.coarse_grained['batch_size'], shuffle=True)

    model = train_coarse_grained_model(train_loader, config)
    save_path = save_model(model, "coarse_grained_model", "models/saved_models")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
