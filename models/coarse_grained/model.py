# models/coarse_grained/model.py
import torch
from models.coarse_grained.components.stochastic_embedding import StochasticEmbeddingModule
from models.coarse_grained.components.transformer_alignment import TransformerAlignment
from models.coarse_grained.components.similarity_calculation import SimilarityCalculation


class CoarseGrainedModel(torch.nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim, output_dim):
        super(CoarseGrainedModel, self).__init__()
        self.video_embedding = StochasticEmbeddingModule(video_dim, hidden_dim)
        self.text_embedding = StochasticEmbeddingModule(text_dim, hidden_dim)
        self.transformer_alignment = TransformerAlignment(hidden_dim)
        self.similarity_calculation = SimilarityCalculation(hidden_dim)

    def forward(self, video_features, text_features):
        video_emb = self.video_embedding.forward(video_features)
        text_emb = self.text_embedding.forward(text_features)
        aligned_emb = self.transformer_alignment(video_emb.unsqueeze(1), text_emb.unsqueeze(1))
        similarity_scores = self.similarity_calculation(aligned_emb.squeeze(1))
        return similarity_scores
