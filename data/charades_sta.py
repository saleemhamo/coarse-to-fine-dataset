import os
import json
import cv2
import numpy as np
from data.datasets import GenericDataset


class CharadesSTA(GenericDataset):
    def __init__(self, video_dir, annotation_file):
        super().__init__(video_dir, annotation_file)
