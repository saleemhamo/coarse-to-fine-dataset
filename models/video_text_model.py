import clip

from .base_model import BaseModel


class VideoTextModel(BaseModel):
    def __init__(self, device):
        super(VideoTextModel, self).__init__()
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def forward(self, video_frames, text):
        video_features = self.clip_model.encode_image(video_frames)
        text_features = self.clip_model.encode_text(text)
        combined = video_features * text_features
        output = combined.sum(dim=-1)  # Example aggregation method
        return output
