import os
import cv2
import numpy as np
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('charades_logger', 'logs/charades.log')


class CharadesSTA:
    def __init__(self, video_dir, train_file, test_file):
        self.video_dir = video_dir
        self.train_file = train_file
        self.test_file = test_file
        logger.info(
            f"Initializing CharadesSTA with video_dir: {video_dir}, train_file: {train_file}, test_file: {test_file}")
        self.train_annotations = self._load_annotations(self.train_file)
        self.test_annotations = self._load_annotations(self.test_file)
        logger.info(
            f"Loaded {len(self.train_annotations)} training annotations and {len(self.test_annotations)} testing annotations.")

    def _load_annotations(self, file_path):
        annotations = []
        logger.info(f"Loading annotations from {file_path}")
        with open(file_path, 'r') as f:
            for line in f.readlines():
                try:
                    video_name, rest = line.strip().split(" ", 1)
                    times, sentence = rest.split("##")
                    start_time, end_time = times.split()
                    annotations.append({
                        "video_name": video_name,
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "sentence": sentence
                    })
                except Exception as e:
                    logger.error(f"Error in line: {line}")
                    logger.error(e)
        logger.info(f"Loaded {len(annotations)} annotations from {file_path}")
        return annotations

    def get_video_path(self, video_name):
        path = os.path.join(self.video_dir, f"{video_name}.mp4")
        logger.info(f"Constructed video path: {path}")
        return path

    def load_video(self, video_name):
        video_path = self.get_video_path(video_name)
        logger.info(f"Loading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from video {video_name}")
        return np.array(frames)

    def get_train_data(self):
        logger.info("Fetching training data")
        return self.train_annotations

    def get_test_data(self):
        logger.info("Fetching testing data")
        return self.test_annotations
