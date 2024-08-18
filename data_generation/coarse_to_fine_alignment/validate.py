import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from dataloaders.tacos_dataloader import TACoSDataset


# Function to calculate Recall@K
def recall_at_k(predictions, labels, k):
    recall = 0
    for i in range(len(labels)):
        if labels[i] in predictions[i][:k]:
            recall += 1
    return recall / len(labels)


# Function to validate model on test data
def validate_model(model, dataloader, k_values=[1, 5, 10]):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            predictions = torch.topk(logits, k=max(k_values), dim=1).indices  # Get top-k predictions
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    recalls = {}
    for k in k_values:
        recalls[f'R@{k}'] = recall_at_k(all_predictions, all_labels, k)

    return recalls


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=tokenizer.vocab_size)

# Load the trained model weights
model.load_state_dict(torch.load('output/final_model.pth'))
model.eval()

# Load the validation/test dataset
test_dataset = TACoSDataset('./data/tacos/tacos.json', './data/tacos/tacos_cg.json', tokenizer, max_len=128)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Validate the model
recalls = validate_model(model, test_dataloader)
print(f"Test/Validation Results: {recalls}")
