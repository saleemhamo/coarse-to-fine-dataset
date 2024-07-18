import torch


class TransformerAlignment(torch.nn.Module):
    def __init__(self, dim, num_layers=6, num_heads=8):
        super(TransformerAlignment, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, video_emb, text_emb):
        memory = self.transformer_encoder(video_emb)
        output = self.transformer_decoder(text_emb, memory)
        return output
