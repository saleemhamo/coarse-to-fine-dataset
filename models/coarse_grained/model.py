import torch
import torch.nn as nn
from models.coarse_grained.helpers import load_clip_model, extract_video_features, extract_text_features, Config


class CoarseGrainedModel(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim):
        super(CoarseGrainedModel, self).__init__()
        self.video_embedding = nn.Linear(video_dim, hidden_dim)
        self.text_embedding = nn.Linear(text_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, video_features, text_features):
        video_emb = self.video_embedding(video_features)
        text_emb = self.text_embedding(text_features)
        output = self.transformer(video_emb.unsqueeze(0), text_emb.unsqueeze(0))
        output = self.fc(output.squeeze(0))  # Ensure output shape is [batch_size]
        return output.squeeze(-1)  # Ensure output shape is [batch_size]


def get_top_k_videos(video_features, text_features, k=5):
    model = CoarseGrainedModel(video_features.size(-1), text_features.size(-1), hidden_dim=512)
    similarity_scores = model(video_features, text_features)
    top_k_indices = similarity_scores.topk(k, dim=1).indices.squeeze(0)
    return top_k_indices


def select_top_k_videos(video_paths, text_query, config, k=5):
    model, processor = load_clip_model(config)
    video_features = torch.stack([extract_video_features(video, model, processor) for video in video_paths])
    text_features = extract_text_features(text_query, model, processor)
    top_k_indices = get_top_k_videos(video_features, text_features, k)
    return [video_paths[i] for i in top_k_indices]
