import os
import torch
import pickle
from transformers import CLIPTokenizer
from model.model_factory import ModelFactory
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict
from stochastic_text_wrapper import StochasticTextWrapper

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


def find_top_k_matches(config, query, model, tokenizer, data_loader, k=5):
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
            video_features = batch['video'].cuda()

            batch_size, num_frames, channels, height, width = video_features.shape
            video_features = video_features.view(batch_size * num_frames, channels, height, width)
            video_features = model.clip.get_image_features(video_features)
            video_features = video_features.view(batch_size, num_frames, -1)

            for trial in range(config.stochasic_trials):
                aligned_text_features, _, _ = model.stochastic(text_features, video_features)

                similarities = torch.matmul(aligned_text_features, video_features.mean(dim=1).t())
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

    # Find the top-k matching videos
    top_videos = find_top_k_matches(config, config.query, model, tokenizer, data_loader, k=5)
    print(f"Top {len(top_videos)} matching videos for the query '{config.query}':")
    for video_id, score in top_videos:
        print(f"Video ID: {video_id}, Score: {score}")


if __name__ == '__main__':
    main()
