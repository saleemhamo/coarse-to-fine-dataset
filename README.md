
# Coarse-to-Fine Grained Text-based Video and Video-moment Retrieval

---

> **Abstract:** *The viral growth of video content across digital platforms has created a significant demand
for effective and accurate video retrieval systems. Traditional approaches, primarily rely on
metadata or keyword-based searches, which puts limitations on handling complex retrieval
tasks that require understanding visual semantics from videos and matching them with specific text queries. This dissertation addresses this challenge by proposing the Coarse-to-Fine
Grained framework for text-based video and video-moment retrieval.
The framework is divided into two stages: a coarse-grained retrieval stage that identifies relevant videos based on general text queries and a fine-grained retrieval stage that pinpoints
specific moments within those videos using more detailed text queries. Leveraging recent advancements in multimodal learning and transformer architectures, particularly the T-MASS
(Text Modeled as a Stochastic Embedding) and MESM (Modal-Enhanced Semantic Modeling) models, the proposed approach aims to enhance retrieval process by integrating these
techniques into a unified pipeline.
The research involved generating coarse-grained dataset annotations from fine-grained annotations of the known TACoS dataset, facilitating the evaluation of the proposed framework
against existing state-of-the-art models. The results demonstrate that the Coarse-to-Fine
Grained framework not only improves retrieval accuracy but also offers a more flexible and
scalable solution for video content analysis.
In conclusion, this dissertation presents a comprehensive solution to the dual challenges of
video and video-moment retrieval, with potential applications ranging from security surveillance to enhancing search functionalities in large video libraries. Future work could explore
further optimization of the framework and its application to broader datasets.*


>
> <p align="center">
> <img width="940" src="GenericPipeline.png">
> </p>


---

## Overview

This project focuses on developing a comprehensive video moment retrieval system that employs a coarse-to-fine-grained text-video alignment strategy. The pipeline integrates methodologies from state-of-the-art research, leveraging techniques such as stochastic embeddings and enhanced semantic modeling to achieve robust retrieval results. The framework is composed of multiple stages that work together to refine the search process, making it both scalable and accurate.

The repository is organized to facilitate the implementation and evaluation of each stage of the pipeline, providing a modular structure that can be extended or modified for further research.

**Research Papers Implemented:**
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
│   └── [README.md](coarse_grained/README.md)     # Instructions for the T-MASS models
│   └── ...
│
├── fine_grained/
│   └── [README.md](fine_grained/README.md)       # Instructions for the MESM model
│   └── ...
│
├── data_generation/                              # Scripts for combined datasets
│   └── tacos_cg                                  # Scripts for generating coarse-grained datasets
│   └── coarse_to_fine_alignment                  # A model to evaluate the generated annotations
│
├── combined_pipeline/                            # Scripts for integrating and running the entire pipeline
│   └── evaluate.py                               # Main script evaluate the entire pipeline
│   └── ...
│
└── ctf_env.yaml                                  # YAML file to create Conda Env
```

## Summary of Achievements

1. **Implemented and Evaluated Stages 1 and 2**: Successfully implemented and evaluated Stage 1 (Coarse-Grained Retrieval) and Stage 2 (Fine-Grained Retrieval) of the framework by leveraging the T-MASS approach for coarse-grained retrieval and the MESM approach for fine-grained retrieval.

2. **Hierarchical Coarse-to-Fine Video and Video-Moment Retrieval Pipeline**: Developed and evaluated a hierarchical pipeline for video and video-moment retrieval, which improves upon traditional text-video retrieval approaches.

3. **Generated Coarse-Grained Annotations**: Created coarse-grained annotations for the TACoS dataset, which involved summarizing fine-grained annotations to facilitate coarse-grained retrieval.

4. **Designed a Model for Annotation Evaluation**: Developed a model to evaluate the generated coarse-grained dataset annotations by attempting to retrieve them based on fine-grained annotations.

5. **Comprehensive Evaluation**: Evaluated each stage of the framework and the combined pipeline using standard metrics, demonstrating the effectiveness of the approach compared to existing state-of-the-art models.

## Hierarchical Evaluation Approach

**Experiment #5: Hierarchical Evaluation Approach**

This experiment aims to evaluate the coarse-to-fine-grained retrieval pipeline in a more suitable way compared to the modular approach. The same datasets and trained models were used, focusing on implementing a hierarchical flow of retrieval and evaluating based on mean and intersection over union metrics, which reflect the nature of the retrieved results.

### Results

| Metric     | Value  |
|------------|--------|
| Mean IoU   | 37.12  |
| Max IoU    | 69.84  |

Results on the combined dataset from TACoS and TACoS-CG in the hierarchical approach reflect the actual performance of the pipeline. The results of Mean IoU show the average overlap between the predicted video segments according to the fine-grained annotations and the ground truth segments of videos retrieved from the coarse-grained stage. The value of 37.21 for Mean IoU is relatively a good indicator that confirms the feasibility of such a framework, while the Max IoU value of 69.84 shows a potential for improving the mean performance of the integrated hierarchical approach.

### Trained Models
[Trained Models Link](https://drive.google.com/drive/folders/1WTfin66IOp3x6cV_A0A8QGIm4aFJfIEl?usp=sharing)

### Generated TACoS-CG Annotations
[Generated TACoS-CG Annotations Link](https://drive.google.com/drive/folders/1FOMKVY5KwfenmU6DILYKZ0FV3eZHd_e9?usp=sharing)


## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to the authors of the T-MASS, MESM, and QD-DETR papers for their foundational work in text-video retrieval.
