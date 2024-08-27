import json
import logging

logger = logging.getLogger(__name__)


def load_tacos_dataset():
    # Load TACoS dataset and TACoS-CG dataset from files
    with open('../../fine_grained/data/TACoS/annotations/test.json') as f:
        fine_grained_data = json.load(f)
    with open('../../coarse_grained/data/TACoS_CG/tacos_cg.json') as f:
        coarse_grained_data = json.load(f)
    with open('../../coarse_grained/data/TACoS_CG/test.txt', 'r') as f:
        coarse_grained_data_test_labels = [line.strip() for line in f]

    filtered_coarse_grained_data = {key: value for key, value in coarse_grained_data.items() if
                                    key in coarse_grained_data_test_labels}

    # Mapping based on video_id
    tacos_dataset = {}
    tacos_cg_dataset = {}
    for coarse_item in filtered_coarse_grained_data:
        video_id = coarse_item["video_id"]
        if video_id in fine_grained_data:
            tacos_dataset[video_id] = fine_grained_data[video_id]
            tacos_cg_dataset[video_id] = coarse_item

    return tacos_dataset, tacos_cg_dataset
