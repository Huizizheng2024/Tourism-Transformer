import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from torchmetrics import Accuracy

class TransformerModel(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, learning_rate: float = 1e-5):
        super(TransformerModel, self).__init__()
        
        # Load pre-trained Transformer model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        
        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        # Forward pass through the Transformer model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the hidden state corresponding to the first token (CLS token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass the CLS token hidden state through the classifier
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def tokenize_batch(self, texts):
        # Tokenize the batch of texts
        return self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
