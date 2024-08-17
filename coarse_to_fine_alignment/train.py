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

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    logging.info(f"Starting Fold {fold + 1}")

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(train_dataset, batch_size=16, sampler=val_sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(3):  # Train for a few epochs
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

            logging.info(f"Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

    # Validation logic
    model.eval()
    total_loss = 0
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

    avg_val_loss = total_loss / len(val_dataloader)
    logging.info(f"Fold {fold + 1} Validation Loss: {avg_val_loss}")

torch.save(model.state_dict(), 'output/final_model.pth')
logging.info("Model training and saving completed.")
