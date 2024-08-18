import json
import torch
import logging
from torch.utils.data import Dataset
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            summaries = defaultdict(list)
            for item in data:
                video_id = item['video_id']
                summaries[video_id].append(item['summarized_sentence'])
            return summaries

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        fine_text = self.fine_annotations[video_id]['sentences']

        # Get all coarse summaries for the given video ID
        coarse_texts = self.coarse_summaries.get(video_id, ["[UNK]"])
        coarse_text = ' '.join(coarse_texts)  # Combine all coarse summaries into one text

        inputs = self.tokenizer.encode_plus(
            fine_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt"
        )

        labels = self.tokenizer.encode_plus(
            coarse_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=False,
            return_tensors="pt"
        )

        if labels['input_ids'].numel() == 0:
            logging.error(f"Empty label tensor for video_id {video_id}, coarse_text: '{coarse_text}'")
            labels['input_ids'] = torch.tensor([self.tokenizer.cls_token_id] * self.max_len)

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()
        }


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    valid_labels = []
    for label in labels:
        if label.dim() != 0:
            valid_labels.append(label)

    if not valid_labels:
        raise ValueError("All labels are invalid.")

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(valid_labels, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
