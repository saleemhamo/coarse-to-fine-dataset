import json

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
        fine_text = self.fine_annotations[video_id]['sentences']
        coarse_text = self.coarse_summaries.get(video_id, "")

        # Log the coarse_text content for debugging
        if not coarse_text:
            logging.warning(f"No coarse summary found for video_id {video_id}. Using a default placeholder.")
            coarse_text = "[UNK]"  # Default placeholder text

        logging.debug(f"Coarse Text for video_id {video_id}: '{coarse_text}'")

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

        # Log the size of the label tensor
        logging.debug(f"Label tensor size for video_id {video_id}: {labels['input_ids'].size()}")

        if labels['input_ids'].numel() == 0:
            logging.error(f"Empty label tensor for video_id {video_id}, coarse_text: '{coarse_text}'")
            return None  # Or handle the empty case appropriately

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()[0]  # Take the first token as the label
        }


import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def collate_fn(batch):
    logging.info("Collate function called.")

    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Log the shape of each sequence in the batch
    for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
        logging.debug(
            f"Sequence {i}: Input ID size = {inp.size()}, Attention Mask size = {attention_mask[i].size()}, Label size = {lbl.size()}")

    # Check for any 0-d tensors (empty sequences)
    if any(seq.dim() == 0 for seq in input_ids) or any(seq.dim() == 0 for seq in labels):
        logging.error("Found a 0-d tensor in the batch. This likely indicates an issue with the data.")
        raise ValueError("Found empty sequence in the batch.")

    # Pad sequences to the max length in this batch
    max_len_input_ids = max([seq.size(0) for seq in input_ids])
    max_len_labels = max([seq.size(0) for seq in labels])
    logging.debug(f"Max length of input IDs in this batch: {max_len_input_ids}")
    logging.debug(f"Max length of labels in this batch: {max_len_labels}")

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    logging.info(
        "Collate function completed successfully. Padded input size: {}, Padded label size: {}".format(input_ids.size(),
                                                                                                       labels.size()))

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
