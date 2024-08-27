import numpy as np
from collections import defaultdict
from fine_grained.utils import compute_temporal_iou_batch_cross, save_json


def calculate_r_at_k(results, k_values):
    metrics = {}
    for k in k_values:
        correct_at_k = sum(1 for result in results if result["rank"] <= k)
        metrics[f"R@{k}"] = correct_at_k / len(results)
    return metrics


def compute_hierarchical_metrics(coarse_results, fine_results):
    combined_metrics = defaultdict(float)
    for coarse, fine in zip(coarse_results, fine_results):
        video_id, coarse_data = coarse
        _, fine_data = fine
        iou = compute_temporal_iou_batch_cross(fine_data["spans"], coarse_data["spans"])
        combined_metrics["Mean_IoU"] += np.mean(iou)
        combined_metrics["Max_IoU"] += np.max(iou)
    for key in combined_metrics:
        combined_metrics[key] /= len(fine_results)
    return combined_metrics


def save_results(metrics, filename):
    save_json(metrics, filename)
