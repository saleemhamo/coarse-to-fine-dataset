import os
import logging
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from dataloaders.tacos_dataloader import TACoSDataset, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=tokenizer.vocab_size).to(device)

train_dataset = TACoSDataset('./data/tacos/tacos.json', './data/tacos/tacos_cg.json', tokenizer, max_len=128)
kf = KFold(n_splits=5)

output_dir = 'output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def recall_at_k(predictions, labels, k):
    recall = 0
    for i in range(len(labels)):
        if labels[i] in predictions[i][:k]:
            recall += 1
    return recall / len(labels)


for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    logging.info(f"Starting Fold {fold + 1}")

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(train_dataset, batch_size=16, sampler=val_sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(3):  # Train for a few epochs
        logging.info(f"Epoch {epoch + 1}/{3}")
        for batch in train_dataloader:
            optimizer.zero_grad()

            fine_input_ids = batch['input_ids'].to(device)
            fine_attention_mask = batch['attention_mask'].to(device)
            coarse_input_ids = batch['labels'].to(device)

            coarse_input_ids = coarse_input_ids.squeeze(1)

            fine_outputs = model(input_ids=fine_input_ids, attention_mask=fine_attention_mask)
            coarse_outputs = model(input_ids=coarse_input_ids,
                                   attention_mask=(coarse_input_ids != tokenizer.pad_token_id).long())

            logits = fine_outputs.logits
            coarse_logits = coarse_outputs.logits

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), coarse_logits.view(-1, model.config.num_labels))

            logging.info(f"Training Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            similarities = torch.matmul(logits, coarse_logits.transpose(0, 1))  # [batch_size, batch_size]
            predictions = torch.topk(similarities, k=5, dim=1).indices  # Get top-5 predictions

            decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            decoded_labels = [tokenizer.decode(coarse_input_ids[i], skip_special_tokens=True) for i in
                              range(coarse_logits.size(0))]

            logging.info(f"Top-5 Predictions: {decoded_predictions}")
            logging.info(f"Ground Truth: {decoded_labels}")

    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            fine_input_ids = batch['input_ids'].to(device)
            fine_attention_mask = batch['attention_mask'].to(device)
            coarse_input_ids = batch['labels'].to(device)

            coarse_input_ids = coarse_input_ids.squeeze(1)

            fine_outputs = model(input_ids=fine_input_ids, attention_mask=fine_attention_mask)
            coarse_outputs = model(input_ids=coarse_input_ids,
                                   attention_mask=(coarse_input_ids != tokenizer.pad_token_id).long())

            logits = fine_outputs.logits
            coarse_logits = coarse_outputs.logits

            loss = loss_fct(logits.view(-1, model.config.num_labels), coarse_logits.view(-1, model.config.num_labels))
            total_loss += loss.item()

            similarities = torch.matmul(logits, coarse_logits.transpose(0, 1))  # [batch_size, batch_size]
            predictions = torch.topk(similarities, k=5, dim=1).indices  # Get top-5 predictions
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(torch.arange(coarse_logits.size(0)).cpu().numpy())  # Ground truth positions

            decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            decoded_labels = [tokenizer.decode(coarse_input_ids[i], skip_special_tokens=True) for i in
                              range(coarse_logits.size(0))]

            logging.info(f"Validation Predictions: {decoded_predictions}")
            logging.info(f"Validation Ground Truth: {decoded_labels}")

    avg_val_loss = total_loss / len(val_dataloader)
    logging.info(f"Fold {fold + 1} Validation Loss: {avg_val_loss}")

    recall_1 = recall_at_k(all_predictions, all_labels, k=1)
    recall_5 = recall_at_k(all_predictions, all_labels, k=5)
    logging.info(f"Fold {fold + 1} Recall@1: {recall_1}")
    logging.info(f"Fold {fold + 1} Recall@5: {recall_5}")

torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
logging.info("Model training and saving completed.")
