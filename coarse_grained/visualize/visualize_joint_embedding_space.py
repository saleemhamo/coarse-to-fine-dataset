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

    # Dimensionality reduction using PCA to 3D
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Separate text and video embeddings
    num_text_embeddings = text_features.shape[0]
    text_embeddings_3d = reduced_embeddings[:num_text_embeddings]
    video_embeddings_3d = reduced_embeddings[num_text_embeddings:]

    # Plotting the embeddings in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(text_embeddings_3d[:, 0], text_embeddings_3d[:, 1], text_embeddings_3d[:, 2], c='r', label='Text Embeddings')
    ax.scatter(video_embeddings_3d[:, 0], video_embeddings_3d[:, 1], video_embeddings_3d[:, 2], c='b', label='Video Embeddings')

    ax.set_title('Joint Embedding Space of Text and Video Embeddings in 3D')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    plt.grid(True)

    # Save the 3D plot as an image
    plt.savefig('joint_embedding_space_3d.png')
    print("3D Plot saved as 'joint_embedding_space_3d.png'")

    # Example of plotting Similarity-Aware Radius Module (R) values
    r_values = video_features_pooled.detach().norm(dim=1).numpy()  # Detach before converting to numpy
    plt.figure(figsize=(10, 7))
    plt.hist(r_values, bins=20, alpha=0.7, color='g')
    plt.title('Histogram of Similarity-Aware Radius (R) Values')
    plt.xlabel('R Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the R values histogram as an image
    plt.savefig('similarity_aware_radius_histogram.png')
    print("Histogram saved as 'similarity_aware_radius_histogram.png'")

if __name__ == "__main__":
    main()
