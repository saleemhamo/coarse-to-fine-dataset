# Coarse-to-Fine Grained Text-based Video-moment Retrieval

## Overview

This project aims to develop a video moment retrieval model that can perform coarse-to-fine-grained text-video alignment. The project is currently under development and leverages techniques from two key research papers:
- **T-MASS**: Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval
  - [Paper](https://arxiv.org/abs/2312.12155v1)
  - [GitHub Repository](https://github.com/Salesforce/T-MASS)
- **MESM**: Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval
  - [Paper](https://arxiv.org/abs/2306.15012)
  - [GitHub Repository](https://github.com/JIYANGGAO/grounded-video-description)

## Tackling the Coarse-to-Fine Issue

To address the coarse-to-fine-grained text-video alignment problem, the project follows these steps:

1. **Data Preparation**
   - Load and preprocess video frames and annotations from datasets such as Charades-STA, TACoS, and QVHighlights.

2. **Feature Extraction**
   - Use pre-trained models (e.g., CLIP) to extract visual and textual features.

3. **Model Training**
   - Train a video-text retrieval model using techniques from the T-MASS and MESM papers.
   - Implement coarse-to-fine alignment strategies to enhance the granularity of the retrieved video moments.

4. **Dataset Generation**
   - Use the trained model to assist in generating fine-grained annotations from coarse ones.

5. **Evaluation**
   - Evaluate the model's performance using appropriate metrics such as Recall@K, Mean Average Precision (mAP), and Intersection over Union (IoU).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to the authors of the T-MASS and MESM papers for their foundational work in text-video retrieval.
