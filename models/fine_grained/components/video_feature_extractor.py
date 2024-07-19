# models/fine_grained/video_feature_extractor.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VideoFeatureExtractorBase:
    def extract_features(self, frame, device):
        raise NotImplementedError


class ResNetVideoFeatureExtractor(VideoFeatureExtractorBase):
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame, device):
        frame = self.transform(frame).to(device)
        frame = frame.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(frame)
        return features.squeeze()


class CLIPVideoFeatureExtractor(VideoFeatureExtractorBase):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_features(self, frame, device):
        inputs = self.processor(images=frame, return_tensors="pt").to(device)
        outputs = self.model.get_image_features(**inputs)
        return outputs
