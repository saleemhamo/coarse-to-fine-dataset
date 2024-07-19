# models/fine_grained/segment_detector.py
import torch
from transformers import QDDETRModel


class SegmentDetector:
    def __init__(self, model_name='qd-detr'):
        self.model = QDDETRModel.from_pretrained(model_name)

    def detect_segments(self, video_features, text_features):
        segments = self.model(video_features, text_features)
        return segments
