# models/fine_grained/components/text_feature_extractor.py
import torch
from transformers import BertModel, CLIPModel, CLIPProcessor


class BERTTextFeatureExtractor:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def extract_features(self, text, device):
        inputs = self.model(text.to(device))
        return inputs.last_hidden_state


class CLIPTextFeatureExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def extract_features(self, text, device):
        inputs = self.processor(text=[text], return_tensors="pt").to(device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features
