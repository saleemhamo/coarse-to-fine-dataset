import torch
import torch.nn as nn


class QDDETRModel(nn.Module):
    def __init__(self, hidden_dim):
        super(QDDETRModel, self).__init__()
        self.transformer = nn.Transformer(hidden_dim, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, video_features, text_features):
        output = self.transformer(video_features.unsqueeze(0), text_features.unsqueeze(0))
        output = self.fc(output.squeeze(0))
        return output
