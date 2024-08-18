import json

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from coarse_grained.config.base_config import Config
import torchvision.transforms as transforms


class TACoSCoarseGrainedDataset(Dataset):
    def __init__(self, config: Config, json_file, split_file, img_transforms=None):
        self.videos_dir = config.videos_dir
        dir = '../data/TACoS_CG'
        json_file = dir + '/' + json_file
        split_file = dir + '/' + split_file

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Load the video IDs from the split file
        with open(split_file, 'r') as f:
            self.video_ids = set(f.read().splitlines())

        # Filter the data based on the video IDs in the split file
        self.data = [entry for entry in self.data if entry['video_id'] in self.video_ids]

        self.img_transforms = img_transforms

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        video_id = entry['video_id']
        summarized_sentence = entry['summarized_sentence']

        # Load video frames
        video_path = f"{self.videos_dir}/{video_id}"
        imgs = self.load_video_frames(video_path)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': summarized_sentence
        }

    def load_video_frames(self, video_path, num_frames=12, video_sample_type='uniform'):  # Ensure num_frames is 12

        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_sample_type == 'uniform':
            # Sample uniformly across the video
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            # Sample random frames
            frame_indices = np.sort(np.random.choice(range(total_frames), num_frames, replace=False))

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = transforms.ToTensor()(frame)
                if self.img_transforms:
                    frame = self.img_transforms(frame)
                frames.append(frame)
            else:
                black_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                black_frame = transforms.ToTensor()(black_frame)
                if self.img_transforms:
                    black_frame = self.img_transforms(black_frame)
                frames.append(black_frame)

        cap.release()
        return torch.stack(frames)
