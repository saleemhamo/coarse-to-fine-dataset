# scripts/train.py

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import clip
from data import get_dataset
from models import VideoTextModel
from utils import get_device, save_model
from utils.constants import CHARADES_STA, MOUNTED_CLAIM_DIRECTORY


def main():
    device = get_device()
    dataset = get_dataset(CHARADES_STA)
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    # Set the cache directory for CLIP
    os.environ['CLIP_CACHE_PATH'] = '/app/cache'

    model = VideoTextModel(device=device).to(device)

    train_videos, train_texts, train_labels = preprocess_data(train_data, model, device)
    test_videos, test_texts, test_labels = preprocess_data(test_data, model, device)

    train_loader = DataLoader(TensorDataset(train_videos, train_texts, train_labels), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_videos, test_texts, test_labels), batch_size=32, shuffle=False)

    train_model(model, train_loader, device)

    models_dir = os.path.join(MOUNTED_CLAIM_DIRECTORY, 'models')
    save_model(model, 'video_text_model', models_dir)


def preprocess_data(data, model, device):
    videos, texts, labels = [], [], []
    for annotation in data:
        video_features = load_video_features(annotation["video_name"], model, device)
        text_features = load_text_features(annotation["sentence"], model, device)
        label = compute_label(annotation["start_time"], annotation["end_time"])
        videos.append(video_features)
        texts.append(text_features)
        labels.append(label)
    return torch.stack(videos), torch.stack(texts), torch.tensor(labels)


def load_video_features(video_name, model, device):
    video_path = os.path.join(model.video_dir, f"{video_name}.mp4")
    video = Image.open(video_path).convert("RGB")
    video = model.clip_preprocess(video).unsqueeze(0).to(device)
    video_features = model.clip_model.encode_image(video)
    return video_features


def load_text_features(sentence, model, device):
    text = clip.tokenize(sentence).to(device)
    text_features = model.clip_model.encode_text(text)
    return text_features


def compute_label(start_time, end_time):
    return torch.tensor([start_time, end_time])


def train_model(model, train_loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(10):
        for video_features, text_features, labels in train_loader:
            video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(
                device)
            optimizer.zero_grad()
            output = model(video_features, text_features)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()
