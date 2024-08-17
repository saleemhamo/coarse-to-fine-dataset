import torch
import torch.nn as nn


class TransformerAlignment(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerAlignment, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        text_embeds: [num_texts, embed_dim] -> [num_texts, 1, embed_dim]
        video_embeds: [num_vids, num_frames, embed_dim] -> [num_vids * num_frames, embed_dim] -> [num_vids * num_frames, 1, embed_dim]
        """
        # Add batch dimension for single instance
        text_embeds = text_embeds.unsqueeze(1)  # [num_texts, 1, embed_dim]
        video_embeds = video_embeds.view(-1, self.embed_dim).unsqueeze(1)  # [num_vids * num_frames, 1, embed_dim]

        # Self-attention on text embeddings
        for layer in self.self_attn_layers:
            text_embeds = layer(text_embeds)  # [num_texts, 1, embed_dim]

        # Cross-attention between text and video embeddings
        for layer in self.cross_attn_layers:
            attn_output, _ = layer(text_embeds, video_embeds, video_embeds)
            text_embeds = self.layer_norm(text_embeds + attn_output)

        # Remove the batch dimension
        text_embeds = text_embeds.squeeze(1)  # [num_texts, embed_dim]

        return text_embeds
