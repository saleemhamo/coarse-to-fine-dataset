# evaluate.py
import torch
from torch.utils.data import DataLoader
from data.charades_sta import CharadesSTA
from models.coarse_grained.components.feature_extractor import FeatureExtractor
from models.coarse_grained.model import CoarseGrainedModel
from utils.config import Config
from utils.model_utils import load_model, get_device
from utils.constants import CHARADES_VIDEOS_DIR, CHARADES_ANNOTATIONS_TEST
from utils.logger import setup_logger
from models.coarse_grained.data_loaders.charades_sta_dataset import CharadesSTADataset  # Updated import
from sklearn.metrics import accuracy_score, f1_score
import argparse

# Setup logger
logger = setup_logger('evaluate_logger')


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for video_features, text_features, labels in test_loader:
            video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(
                device)
            outputs = model(video_features, text_features).squeeze(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    logger.info(f"Accuracy: {accuracy}, F1 Score: {f1}")
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Coarse-Grained Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    args = parser.parse_args()

    logger.info("Loading configuration.")
    config = Config()
    charades_sta = CharadesSTA(
        video_dir=CHARADES_VIDEOS_DIR,
        test_file=CHARADES_ANNOTATIONS_TEST
    )
    annotations = charades_sta.get_test_data()

    # Load feature extractor
    logger.info("Loading feature extractor.")
    feature_extractor = FeatureExtractor()

    dataset = CharadesSTADataset(annotations, CHARADES_VIDEOS_DIR, feature_extractor)
    test_loader = DataLoader(dataset, batch_size=config.coarse_grained['batch_size'], shuffle=False)
    logger.info("Data loader created.")

    device = get_device(logger)
    model = CoarseGrainedModel(video_dim=512, text_dim=512, hidden_dim=512, output_dim=1).to(device)

    # Load trained model weights from argument path
    logger.info(f"Loading model from {args.model_path}")
    load_model(model, args.model_path)

    # Evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
