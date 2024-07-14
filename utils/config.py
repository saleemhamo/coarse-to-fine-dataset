import json
import os


class Config:
    def __init__(self, config_file="config.json"):
        # Construct the full path to the configuration file
        root_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(root_dir, '..', config_file)

        with open(config_path, "r") as f:
            config = json.load(f)
        self.coarse_grained = config["coarse_grained"]
        self.clip_arch = self.coarse_grained["clip_arch"]
        self.frame_extraction_interval = self.coarse_grained["frame_extraction_interval"]
        self.learning_rate = self.coarse_grained["learning_rate"]
        self.batch_size = self.coarse_grained["batch_size"]
        self.num_epochs = self.coarse_grained["num_epochs"]

    def __repr__(self):
        return f"Config(coarse_grained={self.coarse_grained})"
