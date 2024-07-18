import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import os


class FeatureExtractor:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_video_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Process frames and extract features
        video_features = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs).squeeze(0)
            video_features.append(features)

        # Averaging features of all frames
        return torch.stack(video_features).mean(dim=0)

    def extract_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).squeeze(0)
        return text_features
