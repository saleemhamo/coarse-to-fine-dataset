import torch
from torch.utils.data import DataLoader
from data.charades_sta import CharadesSTA
from utils.config import Config
from utils.model_utils import save_model, get_device
from utils.constants import CHARADES_VIDEOS_DIR, CHARADES_ANNOTATIONS_TRAIN, CHARADES_ANNOTATIONS_TEST, \
    COARSE_GRAINED_MODELS_DIR
from utils.logger import setup_logger
from models.coarse_grained.components.feature_extractor import (FeatureExtractor)
from models.coarse_grained.model import CoarseGrainedModel
from models.coarse_grained.data_loaders.charades_sta_dataset import CharadesSTADataset  # Updated import
import os
import re

# Setup logger
logger = setup_logger('train_logger')


def train_coarse_grained_model(train_loader, config):
    device = get_device(logger)
    video_dim = train_loader.dataset[0][0].size(-1)
    text_dim = train_loader.dataset[0][1].size(-1)
    model = CoarseGrainedModel(video_dim, text_dim, hidden_dim=512, output_dim=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.coarse_grained['learning_rate'])

    logger.info(f"Starting training for {config.coarse_grained['num_epochs']} epochs.")
    for epoch in range(config.coarse_grained['num_epochs']):
        model.train()
        total_loss = 0
        for video_features, text_features, labels in train_loader:
            video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(
                device)
            optimizer.zero_grad()
            outputs = model(video_features, text_features).squeeze(-1)  # Adjust the shape of outputs
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
    annotations = annotations[:5]  # For testing purposes

    # Load feature extractor
    logger.info("Loading feature extractor.")
    feature_extractor = FeatureExtractor()

    dataset = CharadesSTADataset(annotations, CHARADES_VIDEOS_DIR, feature_extractor)
    train_loader = DataLoader(dataset, batch_size=config.coarse_grained['batch_size'], shuffle=True)
    logger.info("Data loader created.")

    model = train_coarse_grained_model(train_loader, config)

    model_file_name = "model"

    save_path = save_model(model, COARSE_GRAINED_MODELS_DIR, custom_file_name=model_file_name)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
