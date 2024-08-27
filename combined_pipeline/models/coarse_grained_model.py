import torch
import logging
from coarse_grained.model.model_factory import ModelFactory
from coarse_grained.config.all_config import AllConfig
from fine_grained.utils import merge_state_dict_with_module
from fine_grained.dataset import prepare_batch_input
import numpy as np

logger = logging.getLogger(__name__)


class CoarseGrainedModel:
    def __init__(self):
        self.opt = AllConfig()
        self.model = self.load_model()

    def load_model(self):
        model = ModelFactory.get_model(self.opt)
        logger.info(f"Load checkpoint from {self.opt.resume}")
        checkpoint = torch.load(self.opt.resume, map_location="cpu")
        model_state_dict = merge_state_dict_with_module(
            checkpoint["model"], model.text_encoder.state_dict(), "text_encoder"
        )
        model.load_state_dict(model_state_dict)
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {self.opt.resume}")
        return model

    def retrieve(self, data):
        with torch.no_grad():
            batch = prepare_batch_input(data, self.opt.device, non_blocking=self.opt.pin_memory)
            outputs = self.model(**batch, dataset_name=self.opt.dataset_name, is_training=False)
            prob = torch.softmax(outputs["pred_logits"], -1)
            scores = prob[..., 0]
            pred_spans = outputs["pred_spans"]

            results = {
                "video_id": data["video_id"],
                "spans": pred_spans,
                "scores": scores,
                "correct": self.evaluate_retrieval(pred_spans, data)
            }
        return results

    def evaluate_retrieval(self, pred_spans, data):
        return True if np.random.rand() > 0.5 else False
