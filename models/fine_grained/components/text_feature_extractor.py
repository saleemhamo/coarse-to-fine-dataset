# models/fine_grained/text_feature_extractor.py
import torch
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor


class TextFeatureExtractorBase:
    def extract_features(self, text):
        raise NotImplementedError


class BERTTextFeatureExtractor(TextFeatureExtractorBase):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def extract_features(self, text, device):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class CLIPTextFeatureExtractor(TextFeatureExtractorBase):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_features(self, text, device):
        inputs = self.processor(text=text, return_tensors="pt").to(device)
        outputs = self.model.get_text_features(**inputs)
        return outputs
