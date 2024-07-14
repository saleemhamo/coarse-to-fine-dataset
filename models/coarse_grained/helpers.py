import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.config import Config

config = Config()


def load_clip_model(config):
    if config.coarse_grained['clip_arch'] == 'ViT-B/32':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif config.coarse_grained['clip_arch'] == 'ViT-B/16':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    else:
        raise ValueError("Unsupported CLIP architecture")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def extract_frames(video_path, interval=config.coarse_grained['frame_extraction_interval']):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()
    return frames


def extract_video_features(video_path, model, processor):
    video_features = []
    for frame in extract_frames(video_path):
        inputs = processor(images=Image.fromarray(frame), return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs).squeeze(0)
        video_features.append(features)
    return torch.stack(video_features).mean(dim=0)  # Averaging features of all frames


def extract_text_features(text_query, model, processor):
    inputs = processor(text=text_query, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).squeeze(0)
    return text_features
