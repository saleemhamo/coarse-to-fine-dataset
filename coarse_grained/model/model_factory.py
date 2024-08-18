from coarse_grained.config.base_config import Config
from coarse_grained.model.clip_stochastic import CLIPStochastic

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_stochastic':
            return CLIPStochastic(config)
        else:
            raise NotImplementedError
