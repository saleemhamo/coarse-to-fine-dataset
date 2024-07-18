# Coarse-to-Fine Grained Text-based Video-moment Retrieval

## Overview

This project aims to develop a video moment retrieval model that performs coarse-to-fine-grained text-video alignment. The project is currently under development and leverages techniques from three key research papers:
- **T-MASS**: Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval
  - [Paper](https://arxiv.org/abs/2312.12155v1)
  - [GitHub Repository](https://github.com/Salesforce/T-MASS)
- **MESM**: Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval
  - [Paper](https://arxiv.org/abs/2306.15012)
  - [GitHub Repository](https://github.com/JIYANGGAO/grounded-video-description)
- **QD-DETR**: Query-Dependent DETR for Video Moment Retrieval and Highlight Detection
  - [Paper](https://arxiv.org/abs/2303.13874)
  - [GitHub Repository](https://github.com/wjun0830/QD-DETR)

## Pipeline Overview

To address the coarse-to-fine-grained text-video alignment problem, the project follows a multi-stage pipeline:

1. **Data Preparation**
   - Load and preprocess video frames and annotations from datasets such as Charades-STA, TACoS, and QVHighlights.
   - Ensure the data is formatted correctly for feature extraction and model training.

2. **Feature Extraction**
   - Use pre-trained models like CLIP to extract visual and textual features.
   - Visual features are extracted from video frames, and textual features are extracted from the provided text descriptions.

3. **Coarse-Grained Retrieval (Stage 1)**
   - **Feature Embedding**: Use a stochastic embedding module to transform the extracted features into stochastic embeddings.
   - **Transformer Alignment**: Apply transformer layers to align the video and text embeddings.
   - **Similarity Calculation**: Compute similarity scores between the aligned embeddings to retrieve the top K video segments that are most relevant to the text query.

4. **Fine-Grained Retrieval (Stage 2)**
   - **Feature Enhancement**: Enhance video and text features using techniques from MESM (Modal-Enhanced Semantic Modeling).
   - **Segment Detection**: Use a query-dependent detection model (QD-DETR) to identify the most relevant segments within the top videos retrieved from the coarse-grained stage.
   - **Refinement**: Further refine the search results to improve granularity and accuracy of the retrieved video moments.

5. **Future Work: Dataset Generation**
   - Use the trained model to assist in generating fine-grained annotations from coarse annotations.
   - Enhance existing datasets with more detailed and accurate annotations.

6. **Evaluation**
   - Evaluate the model's performance using metrics such as Recall@K, Mean Average Precision (mAP), and Intersection over Union (IoU).
   - Continuously improve the model based on evaluation results and feedback.

## Tackling the Coarse-to-Fine Issue

The coarse-to-fine issue in text-video alignment is tackled by dividing the problem into two main stages:

### Stage 1: Coarse-Grained Retrieval
- **Feature Extraction using CLIP**: Extract features from videos and text queries using the CLIP model.
- **Stochastic Embedding Module**: Transform video and text embeddings into stochastic embeddings.
- **Transformer Alignment**: Align the embeddings using transformer layers.
- **Similarity Calculation**: Compute similarity scores to identify the top K video segments.

### Stage 2: Fine-Grained Retrieval
- **Feature Enhancement using MESM**: Enhance the features extracted in the coarse-grained stage.
- **Segment Detection using QD-DETR**: Detect relevant segments within the top videos using a query-dependent detection model.
- **Top K Segment Selection**: Refine the search results to select the most relevant video segments.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to the authors of the T-MASS, MESM, and QD-DETR papers for their foundational work in text-video retrieval.
