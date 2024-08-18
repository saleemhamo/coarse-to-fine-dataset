import os
import torch
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import numpy as np

# Imports from your existing codebase
from coarse_grained.config.all_config import AllConfig
from coarse_grained.datasets.data_factory import DataFactory
from coarse_grained.model.model_factory import ModelFactory
from coarse_grained.modules.metrics import t2v_metrics, v2t_metrics
from coarse_grained.modules.loss import LossFactory
from coarse_grained.trainer.trainer_stochastic import Trainer
from fine_grained.runner import build_model, build_dataloader, build_vocab
from fine_grained.utils import TestOptions, PostProcessorDETR, merge_state_dict_with_module, \
    save_json, save_jsonl
from fine_grained.utils import span_cxw_to_xx, compute_temporal_iou_batch_cross, interpolated_precision_recall, \
    temporal_nms
from fine_grained.dataset.base import prepare_batch_input

# Set up logging
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # # Step 1: Coarse-Grained Evaluation
    coarse_grained_config = AllConfig()
    coarse_grained_results = evaluate_coarse_grained(coarse_grained_config)

    # Step 2: Fine-Grained Evaluation
    fine_grained_results = evaluate_fine_grained(coarse_grained_results)

    # Step 3: Compute Unified Metric
    unified_metrics = compute_unified_metrics(coarse_grained_results, fine_grained_results)

    # Step 4: Print Metrics
    print_metrics(unified_metrics)


def evaluate_coarse_grained(config):
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if config.gpu is not None and config.gpu != '99':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # Set random seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model, loss=loss, metrics=metrics, optimizer=None, config=config, train_data_loader=None,
                      valid_data_loader=test_data_loader, lr_scheduler=None, writer=None, tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")

    coarse_grained_results = trainer.validate()
    return coarse_grained_results


def evaluate_fine_grained(coarse_grained_results):
    opt = TestOptions().parse()
    vocab = build_vocab(opt) if opt.tokenizer_type in ["GloVeSimple", "GloVeNLTK"] else None
    _, _, test_loader = build_dataloader(opt, vocab)
    for split in test_loader.keys():
        test_loader = test_loader[split]
        break
    model = build_model(opt, vocab)

    logger.info(f"Load checkpoint from {opt.resume}")
    checkpoint = torch.load(opt.resume, map_location="cpu")
    if model.text_encoder is not None:
        model_state_dict = merge_state_dict_with_module(checkpoint["model"], model.text_encoder.state_dict(),
                                                        "text_encoder")
    else:
        model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict)
    logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")

    with torch.inference_mode():
        fine_grained_results = fine_grained_eval(model, test_loader, opt)

    return fine_grained_results


@torch.no_grad()
def fine_grained_eval(model, eval_loader, opt):
    model.eval()
    metrics = defaultdict(list)

    for batch in tqdm(eval_loader, desc="Evaluating Fine-Grained Stage"):
        prepare_batch_input(batch, opt.device, non_blocking=opt.pin_memory)
        outputs = model(**batch, dataset_name=opt.dataset_name, is_training=False)
        prob = torch.softmax(outputs["pred_logits"], -1)

        spans = outputs["pred_spans"]
        scores = prob[..., 0]

        for idx, (span, score) in enumerate(zip(spans, scores)):
            spans = span_cxw_to_xx(span) * batch["duration"][idx]
            metrics['spans'].append(spans.cpu().numpy())
            metrics['scores'].append(score.cpu().numpy())

    return metrics

def compute_combined_metrics(coarse_grained_results, fine_grained_results):
    combined_metrics = {}

    # Extract necessary data from fine-grained results
    fine_spans = np.array(fine_grained_results['spans'])
    fine_scores = np.array(fine_grained_results['scores'])

    # Compute IoU between fine-grained spans (as a form of alignment verification)
    iou = compute_temporal_iou_batch_cross(fine_spans, fine_spans)

    # Flatten fine-grained scores for mAP calculation
    fine_scores_flat = fine_scores.flatten()

    # Calculate mAP score
    map_score = interpolated_precision_recall(fine_scores_flat, fine_scores_flat)

    # Combine R@K metrics from the coarse-grained stage with the mAP from the fine-grained stage
    combined_metrics['R1_combined'] = (coarse_grained_results['R1'] + map_score) / 2
    combined_metrics['R5_combined'] = (coarse_grained_results['R5'] + map_score) / 2
    combined_metrics['R10_combined'] = (coarse_grained_results['R10'] + map_score) / 2
    combined_metrics['R50_combined'] = (coarse_grained_results['R50'] + map_score) / 2
    combined_metrics['R100_combined'] = (coarse_grained_results['R100'] + map_score) / 2

    # Additional Metrics
    combined_metrics['Mean_IoU'] = np.mean(iou)  # Mean IoU across all fine-grained spans
    combined_metrics['Median_IoU'] = np.median(iou)  # Median IoU
    combined_metrics['Max_IoU'] = np.max(iou)  # Max IoU

    # Displaying the combined metrics
    print("\n--- Combined Metrics ---")
    for key, value in combined_metrics.items():
        print(f"{key}: {value:.4f}")

    return combined_metrics

def print_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
