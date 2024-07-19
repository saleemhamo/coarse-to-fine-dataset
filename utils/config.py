import json
import os


class Config:
    def __init__(self, config_file="config.json"):
        # Construct the full path to the configuration file
        root_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(root_dir, '..', config_file)

        with open(config_path, "r") as f:
            config = json.load(f)

        # Coarse-grained configuration
        self.coarse_grained = config["coarse_grained"]
        self.clip_arch = self.coarse_grained["clip_arch"]
        self.frame_extraction_interval = self.coarse_grained["frame_extraction_interval"]
        self.learning_rate = self.coarse_grained["learning_rate"]
        self.batch_size = self.coarse_grained["batch_size"]
        self.num_epochs = self.coarse_grained["num_epochs"]

        # Fine-grained configuration
        self.fine_grained = config["fine_grained"]
        self.fine_grained_text_extractor = self.fine_grained["text_extractor"]
        self.fine_grained_video_extractor = self.fine_grained["video_extractor"]
        self.fine_grained_learning_rate = self.fine_grained["learning_rate"]
        self.fine_grained_batch_size = self.fine_grained["batch_size"]
        self.fine_grained_num_epochs = self.fine_grained["num_epochs"]

    def __repr__(self):
        return (f"Config(coarse_grained={self.coarse_grained}, "
                f"fine_grained={self.fine_grained})")
