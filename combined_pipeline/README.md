# Hierarchical Video Retrieval Evaluation

This project is designed to evaluate a hierarchical video retrieval system using both coarse-grained and fine-grained retrieval models on the TACoS dataset. The code assesses the effectiveness of retrieving entire videos based on a summarized query and then refining the search to predict specific moments within the video.

## Overview

The hierarchical retrieval system works in two stages:
1. **Coarse-Grained Retrieval**: Identifies the most relevant videos based on a broad description of the content.
2. **Fine-Grained Retrieval**: Narrows down to specific moments within the video based on more detailed annotations.

The evaluation process first retrieves videos using the coarse-grained model. If a video is successfully retrieved within the top-k results, it proceeds to the fine-grained stage where specific moments are predicted. The system's performance is measured using metrics like Recall at k (R@k), Mean Rank (MeanR), Median Rank (MedR), and Mean Intersection over Union (mIoU).

## Project Structure

```plaintext
- evaluation.py
- evaluation_modular.py
- models/
  - coarse_grained_model.py
  - fine_grained_model.py
- datasets/
  - tacos_loader.py
- utils/
  - metrics.py
  - logging_setup.py
- README.md
```
## Why This Code is Important

This evaluation framework is crucial for understanding the effectiveness of a hierarchical retrieval approach, where broad video retrieval is refined to pinpoint specific actions or events within a video. It helps in assessing how well the combination of coarse and fine models performs compared to using either model alone.

## Running the Evaluation

1. **Load the Data**: Ensure that the TACoS and TACoS-CG datasets are available.
2. **Configure the Models**: Adjust the settings in the respective model files (`coarse_grained_model.py` and `fine_grained_model.py`).
3. **Run the Evaluation**: Execute the `evaluation.py` script to start the hierarchical evaluation. Results will be logged and saved automatically.

## Metrics Logged

- **Coarse-Grained Retrieval Metrics**: Measures how well the system retrieves entire videos based on coarse annotations.
- **Fine-Grained Retrieval Metrics**: Measures the accuracy of predicting specific video moments.
- **Hierarchical Metrics**: Combines both stages to evaluate the overall performance of the system.

## Results

Results from the evaluation will include detailed logs and saved metrics, providing insights into the performance of both retrieval stages individually and combined.