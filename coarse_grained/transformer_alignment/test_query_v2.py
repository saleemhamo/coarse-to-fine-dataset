import os
import torch
import pickle
from transformers import CLIPTokenizer
from model.model_factory import ModelFactory
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict
from stochastic_text_wrapper import StochasticTextWrapper  # Import the new wrapper module

CACHE_DIR = "./cache"


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
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Wrap the stochastic text module with the new wrapper
    model.stochastic = StochasticTextWrapper(config)

    return model, tokenizer


def process_query(query, tokenizer):
    """Tokenize the text query."""
    inputs = tokenizer(query, return_tensors="pt")
    return inputs


def load_data(config):
    """Load and preprocess the video data from MSR-VTT dataset."""
    img_transforms = init_transform_dict(config.input_res)
    dataset = MSRVTTDataset(config, split_type='test', img_transforms=img_transforms['clip_test'])
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


def load_cache(cache_file):
    """Load the cache from a file if it exists."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"Loaded cache from {cache_file}")
    else:
        cache = {}
    return cache


def save_cache(cache, cache_file):
    """Save the cache to a file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {cache_file}")


def find_top_k_matches(config, query, model, tokenizer, data_loader, cache, k=5):
    """Find the top-k matching videos for the given query."""
    text_inputs = process_query(query, tokenizer)
    text_features = model.clip.get_text_features(
        input_ids=text_inputs['input_ids'].cuda(),
        attention_mask=text_inputs['attention_mask'].cuda()
    )

    video_scores = {}

    with torch.no_grad():
        for batch in data_loader:
            video_ids = batch['video_id']
            video_features_list = []

            for idx, video_id in enumerate(video_ids):
                if video_id in cache:
                    # Use cached features
                    video_features = cache[video_id].cuda()  # Move to GPU for processing
                else:
                    # Calculate and cache features
                    video_data = batch['video'][idx].unsqueeze(0).cuda()
                    _, num_frames, channels, height, width = video_data.shape

                    if channels != 3:
                        raise ValueError(f"Expected 3 channels (RGB), but got {channels} channels.")

                    # Reshape to [batch_size * num_frames, channels, height, width]
                    video_data = video_data.view(-1, channels, height, width)
                    video_features = model.clip.get_image_features(video_data)
                    video_features = video_features.view(num_frames, -1)

                    # Cache the features
                    cache[video_id] = video_features.cpu()  # Store on CPU

                video_features_list.append(video_features)

            # Stack video features and calculate similarities
            video_features_tensor = torch.stack(video_features_list).squeeze()

            # If video_features_tensor is 3D, aggregate across the third dimension (frames)
            if video_features_tensor.dim() == 3:
                video_features_tensor = video_features_tensor.mean(dim=1)  # Averaging over frames

            for trial in range(config.stochasic_trials):
                text_embed_stochastic, _, _ = model.stochastic(text_features, video_features_tensor)

                similarities = torch.matmul(text_embed_stochastic, video_features_tensor.t())
                top_scores, top_indices = similarities.topk(k, dim=1)

                for score, idx in zip(top_scores.cpu().numpy().flatten(), top_indices.cpu().numpy().flatten()):
                    video_id = video_ids[idx]
                    if video_id in video_scores:
                        video_scores[video_id] = max(video_scores[video_id], score)
                    else:
                        video_scores[video_id] = score

    # Sort video scores and get the top-k
    sorted_videos = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_videos[:k]


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

    # Define cache file path
    cache_file = os.path.join(CACHE_DIR, f"{config.dataset_name}_video_features.pkl")

    # Load existing cache or initialize new one
    cache = load_cache(cache_file)

    # Find the top-k matching videos
    top_videos = find_top_k_matches(config, config.query, model, tokenizer, data_loader, cache, k=5)
    print(f"Top {len(top_videos)} matching videos for the query '{config.query}':")
    for video_id, score in top_videos:
        print(f"Video ID: {video_id}, Score: {score}")

    # Save the updated cache
    save_cache(cache, cache_file)


if __name__ == '__main__':
    main()
