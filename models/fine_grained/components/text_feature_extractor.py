# models/fine_grained/components/text_feature_extractor.py
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor
import torch


class BERTTextFeatureExtractor:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def extract_features(self, text, device):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class CLIPTextFeatureExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_features(self, text, device):
        if isinstance(text, torch.Tensor):
            text = text.tolist()
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.model.get_text_features(**inputs)
        return outputs
