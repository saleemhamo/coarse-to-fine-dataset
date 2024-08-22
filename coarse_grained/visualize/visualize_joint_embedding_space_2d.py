import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from transformers import CLIPTokenizer

# Main function to create and save the plot
def main():
    # Set up configuration
    config = AllConfig()
    dataloader = DataFactory.get_data_loader(config, split_type='train')
    model = ModelFactory.get_model(config)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # Process the first batch to extract embeddings
    for batch in dataloader:
        # Tokenize the text data
        batch['text'] = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        print(batch['text'])
        text_features, video_features_pooled, _ = model(batch, return_all_frames=False, is_train=False)
        break  # Use only the first batch for this example

    # Reduce the dimensionality of video features by averaging across the frames dimension
    video_features_pooled = video_features_pooled.mean(dim=1)

    # Combine text and video embeddings
    embeddings = torch.cat((text_features.detach(), video_features_pooled.detach()), dim=0).numpy()

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Separate text and video embeddings
    num_text_embeddings = text_features.shape[0]
    text_embeddings_2d = reduced_embeddings[:num_text_embeddings]
    video_embeddings_2d = reduced_embeddings[num_text_embeddings:]

    # Plotting the embeddings
    plt.figure(figsize=(10, 7))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], c='r', label='Text Embeddings')
    plt.scatter(video_embeddings_2d[:, 0], video_embeddings_2d[:, 1], c='b', label='Video Embeddings')
    plt.title('Joint Embedding Space of Text and Video Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig('joint_embedding_space.png')
    print("Plot saved as 'joint_embedding_space.png'")

if __name__ == "__main__":
    main()
