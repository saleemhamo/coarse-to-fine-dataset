import logging
from datasets.tacos_loader import load_tacos_dataset
from models.coarse_grained_model import CoarseGrainedModel
from models.fine_grained_model import FineGrainedModel
from utils.metrics import compute_hierarchical_metrics, save_results, calculate_r_at_k
from utils.logging_setup import setup_logging

logger = setup_logging()


def hierarchical_evaluation():
    # Step 1: Load the TACoS Dataset
    logger.info("Loading TACoS Dataset")
    tacos_dataset, tacos_cg_dataset = load_tacos_dataset()

    # Step 2: Initialize Models
    logger.info("Initializing Models")
    coarse_model = CoarseGrainedModel()
    fine_model = FineGrainedModel()

    # Step 3: Perform Coarse-Grained Retrieval
    logger.info("Performing Coarse-Grained Retrieval")
    coarse_results = []
    for video_id, data in tacos_cg_dataset.items():
        coarse_result = coarse_model.retrieve(data)
        coarse_results.append((video_id, coarse_result))

    # Step 4: Perform Fine-Grained Retrieval on Correctly Retrieved Video IDs
    logger.info("Performing Fine-Grained Retrieval")
    fine_results = []
    for video_id, coarse_result in coarse_results:
        if coarse_result['correct']:  # Check if the coarse retrieval is correct
            fine_result = fine_model.retrieve(tacos_dataset[video_id], coarse_result)
            fine_results.append((video_id, fine_result))

    # Step 5: Compute and Log Metrics
    logger.info("Calculating and Logging Metrics")
    hierarchical_metrics = compute_hierarchical_metrics(coarse_results, fine_results)

    # Coarse-grained only metrics
    coarse_metrics = calculate_r_at_k([result for _, result in coarse_results], k_values=[1, 5, 10, 50])
    logger.info(f"Coarse-Grained Metrics: {coarse_metrics}")

    # Fine-grained only metrics
    fine_metrics = calculate_r_at_k([result for _, result in fine_results], k_values=[1, 5, 10, 50])
    logger.info(f"Fine-Grained Metrics: {fine_metrics}")

    # Combined hierarchical metrics
    logger.info(f"Hierarchical Metrics: {hierarchical_metrics}")

    # Step 6: Output Results
    logger.info("Saving Results")
    save_results(hierarchical_metrics, 'hierarchical_metrics.json')
    save_results(coarse_metrics, 'coarse_grained_metrics.json')
    save_results(fine_metrics, 'fine_grained_metrics.json')
    logger.info("Evaluation Completed. Results saved.")


if __name__ == "__main__":
    hierarchical_evaluation()
