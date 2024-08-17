import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from coarse_to_fine_alignment.dataloaders.tacos_dataloader import TACoSDataset


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


# Load the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=tokenizer.vocab_size)

# Load your dataset
train_dataset = TACoSDataset('./data/tacos/tacos.json', './data/tacos/tacos_cg.json', tokenizer, max_len=128)
kf = KFold(n_splits=5)

# Training and Validation Loop with K-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_dataloader = DataLoader(train_dataset, batch_size=16, sampler=val_sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(3):  # Train for a few epochs
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Validation
    recalls = validate_model(model, val_dataloader)
    print(f"Validation Results: {recalls}")

# Save the final model
torch.save(model.state_dict(), 'output/final_model.pth')
