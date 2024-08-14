# Coarse-to-Fine Grained Text-based Video-moment Retrieval

## Overview

This project aims to develop a video moment retrieval model that performs coarse-to-fine-grained text-video alignment. The project is under development and leverages techniques from three key research papers:
- **T-MASS**: Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval
  - [Paper](https://arxiv.org/abs/2312.12155v1)
  - [GitHub Repository](https://github.com/Salesforce/T-MASS)
- **MESM**: Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval
  - [Paper](https://arxiv.org/abs/2306.15012)
  - [GitHub Repository](https://github.com/JIYANGGAO/grounded-video-description)

## Repository Structure

```plaintext
coarse-to-fine-grained-dataset/
│
├── coarse_grained/
│   ├── T-MASS/               # Sub-repository for T-MASS
│   ├── T-MASS-V2/            # Sub-repository for T-MASS-V2
│   ├── retrieve.py           # Script to interact with the T-MASS models
│   └── README.md             # Instructions for the T-MASS models
│
├── fine_grained/
│   ├── MESM/                 # Sub-repository for MESM
│   ├── retrieve.py           # Script to interact with the MESM model
│   └── README.md             # Instructions for the MESM model
│
├── data_generation/          # Scripts for generating coarse-grained datasets
│   └── ...
│
├── evaluation/
│   ├── evaluate.py           # Unified evaluation script
│   └── ...
│
└── main_pipeline.py          # Main script to run the entire pipeline
```


## Summary of Updates:
- **Repository Structure**: Reflects the sub-repositories for T-MASS, T-MASS-V2, and MESM.
- **Generic Pipeline**: Added brief steps for setting up the environment and running the pipeline using Conda.
- **Inner Repository References**: Added references to `README.md` files in the sub-repositories for more detailed instructions.
- **TACoS-CG Dataset Generation**: TACoSDataset has been generated using the script in `data_generation/` 

This structure should help you maintain clarity and organization in your project, making it easier to collaborate and extend the work.

## Summary of Results & Achievements:
- **Result 1**: Lorem.
- **Result 2**: Lorem.
- **Table of metrics:**

## Pipeline Overview

The project follows a multi-stage pipeline to address the coarse-to-fine-grained text-video alignment problem:

1. **Data Preparation**
   - Load and preprocess video frames and annotations from datasets such as Charades-STA, TACoS, and QVHighlights.

2. **Feature Extraction**
   - Use pre-trained models like CLIP to extract visual and textual features from video frames and text descriptions.

3. **Coarse-Grained Retrieval (Stage 1)**
   - **Feature Embedding**: Transform features into stochastic embeddings using the T-MASS models.
   - **Transformer Alignment**: Align the video and text embeddings using transformer layers.
   - **Similarity Calculation**: Retrieve the top K video segments most relevant to the text query.

4. **Fine-Grained Retrieval (Stage 2)**
   - **Feature Enhancement**: Enhance video and text features using MESM.
   - **Segment Detection**: Identify the most relevant segments within the top videos using a query-dependent detection model (QD-DETR).
   - **Refinement**: Further refine the search results to improve the granularity and accuracy of the retrieved video moments.

5. **Evaluation**
   - Evaluate the model's performance using metrics such as Recall@K, Mean Average Precision (mAP), and Intersection over Union (IoU).

## Running the Pipeline

To run the pipeline, follow these steps:

### 1. **Set Up the Environment**
   - Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you don't have it already.
   - Create a Conda environment and install dependencies:
     ```bash
     conda create -n coarse_to_fine_env python=3.8
     conda activate coarse_to_fine_env
     pip install -r requirements.txt
     ```

### 2. **Clone the Repositories**
   - Ensure that all sub-repositories are initialized:
     ```bash
     git submodule update --init --recursive
     ```

### 3. **Run the Pipeline**
   - Run the main pipeline:
     ```bash
     python main_pipeline.py
     ```

### 4. **Evaluation**
   - Evaluate the entire model:
     ```bash
     python evaluation/evaluate.py
     ```

## References for Inner Repositories

Each of the inner repositories has its own `README.md` file that provides more detailed instructions specific to that part of the project:

- **T-MASS and T-MASS-V2**: See `coarse_grained/T-MASS/README.md` and `coarse_grained/T-MASS-V2/README.md`.
- **MESM**: See `fine_grained/MESM/README.md`.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to the authors of the T-MASS, MESM, and QD-DETR papers for their foundational work in text-video retrieval.