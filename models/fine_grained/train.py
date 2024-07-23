# models/fine_grained/train.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from data.charades_sta import CharadesSTA
from models.coarse_grained.helpers import config
from utils.config import Config
from utils.model_utils import save_model, get_device
from utils.constants import CHARADES_VIDEOS_DIR, CHARADES_ANNOTATIONS_TRAIN, CHARADES_ANNOTATIONS_TEST, \
    FINE_GRAINED_MODELS_DIR
from utils.logger import setup_logger
from models.fine_grained.components.text_feature_extractor import BERTTextFeatureExtractor, CLIPTextFeatureExtractor
from models.fine_grained.components.video_feature_extractor import ResNetVideoFeatureExtractor, \
    CLIPVideoFeatureExtractor
from models.fine_grained.components.cross_attention import CrossAttentionLayer
from models.fine_grained.components.supervision import SupervisionLoss
from models.fine_grained.components.qd_detr import QDDETRModel
from models.fine_grained.data_loaders.charades_sta_dataset import CharadesSTADatasetFineGrained
from transformers import BertTokenizer, CLIPProcessor
import os

# Setup logger
logger = setup_logger('train_logger')

# Initialize tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def collate_fn(batch):
    video_frames, text_sentences, labels = zip(*batch)

    # Pad video frames
    video_frames_padded = pad_sequence([torch.tensor(np.array(v)) for v in video_frames], batch_first=True)

    # Tokenize text sentences
    if config.fine_grained_text_extractor == 'bert':
        text_sentences = [bert_tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0) for text in text_sentences]
    elif config.fine_grained_text_extractor == 'clip':
        text_sentences = [clip_processor(text=[text], return_tensors='pt')['input_ids'].squeeze(0) for text in
                          text_sentences]
    else:
        raise ValueError("Invalid text_extractor value in config")

    # Pad text sentences
    text_sentences_padded = pad_sequence(text_sentences, batch_first=True)

    labels = torch.tensor(labels)
    return video_frames_padded, text_sentences_padded, labels


def fine_grained_retrieval(train_loader, config):
    device = get_device(logger)

    # Choose text feature extractor
    if config.fine_grained_text_extractor == 'bert':
        text_extractor = BERTTextFeatureExtractor()
    elif config.fine_grained_text_extractor == 'clip':
        text_extractor = CLIPTextFeatureExtractor()
    else:
        raise ValueError("Invalid text_extractor value in config")

    # Choose video feature extractor
    if config.fine_grained_video_extractor == 'resnet':
        video_extractor = ResNetVideoFeatureExtractor()
    elif config.fine_grained_video_extractor == 'clip':
        video_extractor = CLIPVideoFeatureExtractor()
    else:
        raise ValueError("Invalid video_extractor value in config")

    cross_attention = CrossAttentionLayer(hidden_dim=512).to(device)
    supervision_loss = SupervisionLoss().to(device)
    detector = QDDETRModel(hidden_dim=512).to(device)

    optimizer = torch.optim.Adam(list(cross_attention.parameters()) + list(detector.parameters()),
                                 lr=config.fine_grained_learning_rate)

    logger.info("Starting fine-grained retrieval.")
    for epoch in range(config.fine_grained_num_epochs):
        total_loss = 0
        for video_frames, text_sentence, labels in train_loader:
            video_frames, text_sentence, labels = video_frames.to(device), text_sentence.to(device), labels.to(device)

            # Extract features
            enhanced_text_features = text_extractor.extract_features(text_sentence, device)
            enhanced_video_features = video_extractor.extract_features(video_frames, device)

            # Cross-Attention
            text_with_video = cross_attention(enhanced_text_features, enhanced_video_features, enhanced_video_features)

            # Supervision Loss
            loss = supervision_loss(text_with_video, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{config.fine_grained_num_epochs}, Loss: {total_loss / len(train_loader)}")

    logger.info("Fine-grained retrieval completed.")
    return detector


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

    dataset = CharadesSTADatasetFineGrained(annotations, CHARADES_VIDEOS_DIR)
    train_loader = DataLoader(dataset, batch_size=config.fine_grained_batch_size, shuffle=True, collate_fn=collate_fn)
    logger.info("Data loader created.")

    model = fine_grained_retrieval(train_loader, config)

    model_file_name = f"model"

    save_path = save_model(model, FINE_GRAINED_MODELS_DIR, custom_file_name=model_file_name)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
