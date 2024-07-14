import unittest

from models.coarse_grained.helpers import extract_text_features, extract_video_features, load_clip_model
from models.coarse_grained.model import select_top_k_videos
from utils.config import Config


class TestCoarseGrainedModel(unittest.TestCase):
    def setUp(self):
        self.video_paths = ["path/to/video1.mp4", "path/to/video2.mp4"]
        self.text_query = "sample text query"
        self.config = Config()

    def test_feature_extraction(self):
        model, processor = load_clip_model(self.config)
        video_features = [extract_video_features(video, model, processor) for video in self.video_paths]
        text_features = extract_text_features(self.text_query, model, processor)
        self.assertEqual(len(video_features), len(self.video_paths))
        self.assertEqual(text_features.size(0), model.config.projection_dim)

    def test_similarity_computation(self):
        top_k_videos = select_top_k_videos(self.video_paths, self.text_query, self.config, k=1)
        self.assertEqual(len(top_k_videos), 1)


if __name__ == "__main__":
    unittest.main()
