import torch
import logging
from fine_grained.runner import build_model as build_fine_grained_model
from fine_grained.utils import merge_state_dict_with_module

logger = logging.getLogger(__name__)


class FineGrainedModel:
    def __init__(self):

        self.opt = build_fine_grained_model()
        self.model = self.load_model()

    def load_model(self):
        model = build_fine_grained_model(self.opt)
        logger.info(f"Load checkpoint from {self.opt.resume}")
        checkpoint = torch.load(self.opt.resume, map_location="cpu")
        model_state_dict = merge_state_dict_with_module(
            checkpoint["model"], model.text_encoder.state_dict(), "text_encoder"
        )
        model.load_state_dict(model_state_dict)
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {self.opt.resume}")
        return model

    def retrieve(self, fine_grained_data, coarse_result):
        # Fine-grained retrieval logic
        with torch.no_grad():
            pred_spans = coarse_result['spans']
            # Using coarse spans to narrow down the search space
            outputs = self.model(pred_spans)
            fine_result = {
                "video_id": coarse_result["video_id"],
                "spans": outputs["pred_spans"],
                "scores": outputs["scores"],
                "correct": self.evaluate_retrieval(fine_grained_data["timestamps"], outputs["pred_spans"])
            }
        return fine_result

    def evaluate_retrieval(self, true_spans, predicted_spans):
        # Simple evaluation logic for matching predicted spans to true spans
        iou_scores = []
        for pred in predicted_spans:
            iou = max([self.compute_iou(pred, true) for true in true_spans])
            iou_scores.append(iou)

        return max(iou_scores) > 0.5  # Example threshold

    def compute_iou(self, span_a, span_b):
        start_a, end_a = span_a
        start_b, end_b = span_b
        inter_start = max(start_a, start_b)
        inter_end = min(end_a, end_b)
        inter_len = max(0, inter_end - inter_start)
        union_len = (end_a - start_a) + (end_b - start_b) - inter_len
        return inter_len / union_len
