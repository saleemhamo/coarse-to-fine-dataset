import os
import torch
import pandas as pd
import pickle
from transformers import CLIPTokenizer
from model.model_factory import ModelFactory
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict
from stochastic_text_wrapper import StochasticTextWrapper
from tqdm import tqdm  # For progress tracking

# Setup logging
import logging

logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

CACHE_FILE = 'video_features_cache.pkl'
NUM_WORKERS = 24  # Set to 75% of available cores


def load_model(config):
    """Load the trained model and tokenizer."""
    model = ModelFactory.get_model(config)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    if config.load_epoch is not None:
        checkpoint_path = os.path.join(config.model_path, f"checkpoint-epoch{config.load_epoch}.pth")
        if config.load_epoch == 0:
            checkpoint_path = os.path.join(config.model_path, "model_best.pth")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Wrap the stochastic text module with the new wrapper
    model.stochastic = StochasticTextWrapper(config)

    return model, tokenizer


def process_query(query, tokenizer):
    """Tokenize the text query."""
    inputs = tokenizer(query, return_tensors="pt").to('cuda')
    return inputs


def load_data(config):
    """Load and preprocess the video data from MSR-VTT dataset."""
    img_transforms = init_transform_dict(config.input_res)
    dataset = MSRVTTDataset(config, split_type='test', img_transforms=img_transforms['clip_test'])
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    return data_loader


def save_cache(cache, file_path):
    """Save the cache to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)


def load_cache(file_path):
    """Load the cache from a file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}


def cache_video_features(config, model, data_loader):
    """Pre-cache all video features."""
    video_features_cache = {}
    model.clip.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Caching video features"):
            video_ids = batch['video_id']
            video_features = batch['video'].cuda()

            if video_features.dim() == 5:
                batch_size, num_frames, channels, height, width = video_features.shape
                video_features_batch = video_features.view(batch_size * num_frames, channels, height, width)
            elif video_features.dim() == 4:
                batch_size, channels, height, width = video_features.shape
                num_frames = 1
                video_features_batch = video_features
            else:
                raise ValueError(f"Unexpected video features shape: {video_features.shape}")

            video_features_clip = model.clip.get_image_features(video_features_batch)
            if num_frames > 1:
                video_features_clip = video_features_clip.view(batch_size, num_frames, -1)

            for idx, video_id in enumerate(video_ids):
                video_features_cache[video_id] = video_features_clip[idx].cpu()

    return video_features_cache


def find_top_k_matches(config, query, model, tokenizer, video_features_cache, k=10):
    """Find the top-k matching videos for the given query."""
    text_inputs = process_query(query, tokenizer)
    text_features = model.clip.get_text_features(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )

    video_scores = {}

    with torch.no_grad():
        for video_id, video_data in video_features_cache.items():
            video_data = video_data.cuda()
            for trial in range(config.stochasic_trials):
                aligned_text_features, _, _ = model.stochastic(text_features, video_data)

                # Debug: print shapes before matrix multiplication
                # print(f"text_features shape: {text_features.shape}")
                # print(f"video_data shape: {video_data.shape}")
                # print(f"aligned_text_features shape: {aligned_text_features.shape}")

                # Ensure the dimensions are correct for matrix multiplication
                video_data_mean = video_data.mean(dim=0)  # shape: [num_frames, embed_dim]
                if video_data_mean.dim() == 2:
                    video_data_mean_2d = video_data_mean.mean(dim=0).unsqueeze(0)  # shape: [1, embed_dim]
                else:
                    video_data_mean_2d = video_data_mean.unsqueeze(0)  # shape: [1, embed_dim]
                aligned_text_features_2d = aligned_text_features.squeeze(0)  # Ensure aligned_text_features is 2D

                # Debug: print shapes before matrix multiplication
                # print(f"aligned_text_features_2d shape: {aligned_text_features_2d.shape}")
                # print(f"video_data_mean_2d shape: {video_data_mean_2d.shape}")

                similarities = torch.matmul(aligned_text_features_2d, video_data_mean_2d.t())

                if similarities.dim() == 1:
                    similarities = similarities.unsqueeze(0)

                available_k = min(k, similarities.shape[1])
                top_scores, top_indices = similarities.topk(available_k, dim=1)

                for score, idx in zip(top_scores.cpu().numpy().flatten(), top_indices.cpu().numpy().flatten()):
                    if video_id in video_scores:
                        video_scores[video_id] = max(video_scores[video_id], score)
                    else:
                        video_scores[video_id] = score

    # Sort video scores and get the top-k
    sorted_videos = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_videos[:k]


def evaluate_model_on_test_data(config, model, tokenizer, test_data, video_features_cache, top_k_values=[10, 50, 100],
                                limit=None):
    """Evaluate the model on test data."""
    max_k = max(top_k_values)
    correct_at_k = [0] * max_k
    ranks = []
    total_queries = len(test_data) if limit is None else limit

    for i, (_, row) in tqdm(enumerate(test_data.iterrows()), total=total_queries, desc="Evaluating queries"):
        if limit is not None and i >= limit:
            break
        query = row['sentence']
        correct_video_id = row['video_id']
        top_videos = find_top_k_matches(config, query, model, tokenizer, video_features_cache, max_k)
        top_video_ids = [video_id for video_id, _ in top_videos]

        for rank, video_id in enumerate(top_video_ids):
            if video_id == correct_video_id:
                ranks.append(rank + 1)
                for j in range(rank, max_k):
                    correct_at_k[j] += 1
                break
        else:
            ranks.append(max_k + 1)

        if i % 10 == 0:  # Log progress every 10 queries
            logger.info(f"Processed {i}/{total_queries} queries")

    recall_at_k = {k: correct_at_k[k - 1] / total_queries for k in top_k_values}
    median_rank = torch.median(torch.tensor(ranks, dtype=torch.float)).item()
    mean_rank = torch.mean(torch.tensor(ranks, dtype=torch.float)).item()

    results = {
        f"R@top{k}": recall_at_k[k] for k in top_k_values
    }
    results.update({
        "MdR": median_rank,
        "MnR": mean_rank
    })

    for metric, value in results.items():
        logger.info(f"{metric}: {value}")
        print(f"{metric}: {value}")

    save_cache(video_features_cache, CACHE_FILE)
    return results


def main():
    # Load configuration from AllConfig, which parses command-line arguments
    config = AllConfig()

    # Set the model path based on the parsed arguments
    config.model_path = os.path.join(config.output_dir, config.exp_name, config.datetime)

    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(config)
    model.to(device)

    # Load data
    data_loader = load_data(config)

    # Cache video features
    video_features_cache = cache_video_features(config, model, data_loader)
    save_cache(video_features_cache, CACHE_FILE)

    # Load test data
    test_data = pd.read_csv('../T-MASS/data/MSRVTT/MSRVTT_JSFUSION_test.csv', names=['key', 'vid_key', 'video_id', 'sentence'],
                            skiprows=1)

    # Evaluate model on test data with a limit of 20 records
    evaluate_model_on_test_data(config, model, tokenizer, test_data, video_features_cache, limit=10)


if __name__ == '__main__':
    main()
