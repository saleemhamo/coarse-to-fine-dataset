import json
import os
from torch.utils.data import Dataset


class TACoSDataset(Dataset):
    def __init__(self, fine_annotations_path, coarse_summaries_path, tokenizer, max_len):
        self.fine_annotations = self.load_fine_annotations(fine_annotations_path)
        self.coarse_summaries = self.load_coarse_summaries(coarse_summaries_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.video_ids = list(self.fine_annotations.keys())

    def load_fine_annotations(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_coarse_summaries(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            summaries = {}
            for item in data:
                video_id = item['video_id']
                summaries[video_id] = item['summarized_sentence']
            return summaries

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        fine_texts = " ".join(self.fine_annotations[video_id]['sentences'])
        coarse_text = self.coarse_summaries[video_id]
        inputs = self.tokenizer(
            fine_texts,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            coarse_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss computation

        return {
            'input_ids': inputs.input_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'labels': labels.flatten(),
        }


# Utility function to load data paths
def load_data_paths():
    fine_file_path = os.path.join("../data", 'tacos', "tacos.json")
    coarse_file_path = os.path.join("../data", 'tacos', "tacos_cg.json")
    return fine_file_path, coarse_file_path
