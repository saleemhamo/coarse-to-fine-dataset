import torch.nn as nn
from modules.stochastic_module import StochasticText
from transformer_alignment import TransformerAlignment


class StochasticTextWrapper(nn.Module):
    def __init__(self, config):
        super(StochasticTextWrapper, self).__init__()
        self.config = config
        self.stochastic = StochasticText(config)
        self.transformer_alignment = TransformerAlignment(
            embed_dim=config.embed_dim,
            num_heads=config.num_mha_heads,
            num_layers=config.num_layers,
            dropout=config.transformer_dropout
        )

    def forward(self, text_features, video_features):
        # Perform stochastic text embedding
        text_embed_stochastic, text_mean, log_var = self.stochastic(text_features, video_features)

        # Align embeddings using transformer layers
        aligned_text_features = self.transformer_alignment(text_embed_stochastic, video_features)

        return aligned_text_features, text_mean, log_var
