import os
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import KFold
from torch.nn import CosineSimilarity

from dataloaders.tacos_dataloader import TACoSDataset, collate_fn

# Set up logging to a file
log_file = 'training.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

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

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    cosine_similarity = CosineSimilarity(dim=1)

    model.train()
    for epoch in range(5):
        logging.info(f"Epoch {epoch + 1}/{5}")
        for batch in train_dataloader:
            optimizer.zero_grad()

            fine_input_ids = batch['input_ids'].to(device)
            fine_attention_mask = batch['attention_mask'].to(device)
            coarse_input_ids = batch['labels'].to(device)
            coarse_input_ids = coarse_input_ids.squeeze(1)

            fine_outputs = model(input_ids=fine_input_ids, attention_mask=fine_attention_mask)
            coarse_outputs = model(input_ids=coarse_input_ids,
                                   attention_mask=(coarse_input_ids != tokenizer.pad_token_id).long())

            fine_logits = fine_outputs.logits
            coarse_logits = coarse_outputs.logits

            loss = -cosine_similarity(fine_logits, coarse_logits).mean()  # Contrastive loss

            logging.info(f"Training Loss: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

            fine_logits = fine_outputs.logits
            coarse_logits = coarse_outputs.logits

            loss = -cosine_similarity(fine_logits, coarse_logits).mean()
            total_loss += loss.item()

            similarities = torch.matmul(fine_logits, coarse_logits.transpose(0, 1))  # [batch_size, batch_size]
            predictions = torch.topk(similarities, k=100, dim=1).indices
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(torch.arange(coarse_logits.size(0)).cpu().numpy())

            decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            decoded_labels = [tokenizer.decode(coarse_input_ids[i], skip_special_tokens=True) for i in
                              range(coarse_logits.size(0))]

            logging.info(f"Validation Predictions: {decoded_predictions}")
            logging.info(f"Validation Ground Truth: {decoded_labels}")

    avg_val_loss = total_loss / len(val_dataloader)
    logging.info(f"Fold {fold + 1} Validation Loss: {avg_val_loss}")

    recall_1 = recall_at_k(all_predictions, all_labels, k=1)
    recall_5 = recall_at_k(all_predictions, all_labels, k=5)
    recall_10 = recall_at_k(all_predictions, all_labels, k=10)
    recall_50 = recall_at_k(all_predictions, all_labels, k=50)
    recall_100 = recall_at_k(all_predictions, all_labels, k=100)
    logging.info(f"Fold {fold + 1} Recall@1: {recall_1}")
    logging.info(f"Fold {fold + 1} Recall@5: {recall_5}")
    logging.info(f"Fold {fold + 1} Recall@10: {recall_10}")
    logging.info(f"Fold {fold + 1} Recall@50: {recall_50}")
    logging.info(f"Fold {fold + 1} Recall@100: {recall_100}")

torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
logging.info("Model training and saving completed.")
