import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint

# Define a PyTorch Dataset class
class EconomicDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            row['text'],  # Assume the data has a 'text' column for economic news or indicators
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(row['label'], dtype=torch.long)  # Assume a 'label' column
        }

# Define the LightningModule for the Transformer model
class EconomicPredictor(pl.LightningModule):
    def __init__(self, model_name, num_labels):
        super(EconomicPredictor, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        val_loss = outputs.loss
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

# Data loading and processing
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_data_loaders(df, tokenizer, batch_size=32, max_length=128):
    train_df, val_df = train_test_split(df, test_size=0.2)
    
    train_dataset = EconomicDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = EconomicDataset(val_df, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def main():
    # Load data
    df = load_data('data/economic_data.csv')

    # Define model parameters
    model_name = "bert-base-uncased"  # Transformer model, you can choose others like roberta or distilbert
    num_labels = 2  # Adjust according to your classification problem

    # Instantiate tokenizer and create data loaders
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader = create_data_loaders(df, tokenizer)

    # Define the Lightning model
    model = EconomicPredictor(model_name, num_labels)

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='economic-predictor-{epoch:02d}-{val_loss:.2f}'
    )

    # Define a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=1, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
