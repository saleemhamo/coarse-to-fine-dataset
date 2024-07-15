import torch
import torch.nn as nn
from models.coarse_grained.helpers import load_clip_model, extract_video_features, extract_text_features, Config
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('coarse_grained_model_logger', 'logs/coarse_grained_model.log')


class CoarseGrainedModel(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim):
        super(CoarseGrainedModel, self).__init__()
        self.video_embedding = nn.Linear(video_dim, hidden_dim)
        self.text_embedding = nn.Linear(text_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(hidden_dim, 1)
        logger.info(
            f"Initialized CoarseGrainedModel with video_dim={video_dim}, text_dim={text_dim}, hidden_dim={hidden_dim}")

    def forward(self, video_features, text_features):
        video_emb = self.video_embedding(video_features)
        text_emb = self.text_embedding(text_features)
        logger.info(f"Video embedding shape: {video_emb.shape}, Text embedding shape: {text_emb.shape}")

        output = self.transformer(video_emb.unsqueeze(0), text_emb.unsqueeze(0))
        logger.info(f"Transformer output shape: {output.shape}")

        output = self.fc(output.squeeze(0))  # Ensure output shape is [batch_size]
        logger.info(f"Fully connected layer output shape: {output.shape}")

        return output.squeeze(-1)  # Ensure output shape is [batch_size]


def get_top_k_videos(video_features, text_features, k=5):
    model = CoarseGrainedModel(video_features.size(-1), text_features.size(-1), hidden_dim=512)
    similarity_scores = model(video_features, text_features)
    logger.info(f"Similarity scores shape: {similarity_scores.shape}")

    top_k_indices = similarity_scores.topk(k, dim=1).indices.squeeze(0)
    logger.info(f"Top {k} video indices: {top_k_indices.tolist()}")

    return top_k_indices


def select_top_k_videos(video_paths, text_query, config, k=5):
    logger.info("Loading CLIP model and processor.")
    model, processor = load_clip_model(config)

    logger.info("Extracting video features.")
    video_features = torch.stack([extract_video_features(video, model, processor) for video in video_paths])
    logger.info(f"Extracted video features shape: {video_features.shape}")

    logger.info("Extracting text features.")
    text_features = extract_text_features(text_query, model, processor)
    logger.info(f"Extracted text features shape: {text_features.shape}")

    top_k_indices = get_top_k_videos(video_features, text_features, k)
    logger.info(f"Selected top {k} videos: {[video_paths[i] for i in top_k_indices.tolist()]}")

    return [video_paths[i] for i in top_k_indices]
