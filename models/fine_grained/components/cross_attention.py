import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output
