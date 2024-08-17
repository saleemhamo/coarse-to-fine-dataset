import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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
        video_id = self.video_ids[idx]  # Use the video ID from the list
        fine_text = " ".join(self.fine_annotations[video_id]['sentences'])  # Combine sentences if needed
        coarse_text = self.coarse_summaries[video_id]  # Get the corresponding coarse summary

        inputs = self.tokenizer.encode_plus(
            fine_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        labels = self.tokenizer.encode_plus(
            coarse_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()[0],  # Take the first token as the label
        }


import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Check if all sequences are non-empty
    if any(len(seq) == 0 for seq in input_ids) or any(len(seq) == 0 for seq in labels):
        raise ValueError("Found empty sequence in the batch.")

    # Pad sequences to the max length in this batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
