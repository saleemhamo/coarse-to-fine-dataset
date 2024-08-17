import torch
import json
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from dataloaders.tacos_dataloader import TACoSDataset, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT tokenizer and the saved model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=tokenizer.vocab_size)
model.load_state_dict(torch.load('output/final_model.pth'))
model.to(device)
model.eval()

# Load your dataset
test_dataset = TACoSDataset('./data/tacos/tacos.json', './data/tacos/tacos_cg.json', tokenizer, max_len=128)
test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

def recall_at_k(predictions, labels, k):
    recall = 0
    for i in range(len(labels)):
        if labels[i] in predictions[i][:k]:
            recall += 1
    return recall / len(labels)

# Evaluation function
def evaluate(model, dataloader, k_values=[1, 5, 10]):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            fine_input_ids = batch['input_ids'].to(device)
            fine_attention_mask = batch['attention_mask'].to(device)
            coarse_input_ids = batch['labels'].to(device)

            coarse_input_ids = coarse_input_ids.squeeze(1)

            fine_outputs = model(input_ids=fine_input_ids, attention_mask=fine_attention_mask)
            coarse_outputs = model(input_ids=coarse_input_ids,
                                   attention_mask=(coarse_input_ids != tokenizer.pad_token_id).long())

            logits = fine_outputs.logits
            coarse_logits = coarse_outputs.logits

            # Calculate similarities
            similarities = torch.matmul(logits, coarse_logits.transpose(0, 1))  # [batch_size, batch_size]
            predictions = torch.topk(similarities, k=max(k_values), dim=1).indices  # Get top-k predictions

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(torch.arange(coarse_logits.size(0)).cpu().numpy())  # Ground truth positions

    recalls = {}
    for k in k_values:
        recalls[f'R@{k}'] = recall_at_k(all_predictions, all_labels, k)

    return recalls

# Run evaluation
logging.info("Starting evaluation...")
recalls = evaluate(model, test_dataloader)
logging.info(f"Evaluation Results: {recalls}")
