import os
import cv2
import numpy as np


class CharadesSTA:
    def __init__(self, video_dir, train_file, test_file):
        self.video_dir = video_dir
        self.train_file = train_file
        self.test_file = test_file
        self.train_annotations = self._load_annotations(self.train_file)
        self.test_annotations = self._load_annotations(self.test_file)

    def _load_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                print(line)
                video_name, rest = line.strip().split(" ", 1)
                times, sentence = rest.split("##")
                start_time, end_time = times.split()
                annotations.append({
                    "video_name": video_name,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "sentence": sentence
                })
        return annotations

    def get_video_path(self, video_name):
        return os.path.join(self.video_dir, f"{video_name}.mp4")

    def load_video(self, video_name):
        video_path = self.get_video_path(video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def get_train_data(self):
        return self.train_annotations

    def get_test_data(self):
        return self.test_annotations
