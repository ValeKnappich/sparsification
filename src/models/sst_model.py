import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from transformers import AutoModel


class SSTModel(pl.LightningModule):
    
    def __init__(self, model_id, lr, out_features=768):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
        self.model_id = model_id
        self.lr = lr
               
        self.bert = AutoModel.from_pretrained(model_id)
        self.head = nn.Linear(out_features, 1)
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
               
        
    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return torch.sigmoid(self.head(features)).reshape(-1)
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=1e-3)
    
    
    def step(self, batch, split="train"):
        y_hat = self.forward(batch)
        loss = .66 * self.mse(batch["label"], y_hat) + .33 * self.mae(batch["label"], y_hat)
        # loss = self.bce(batch["label"], y_hat) + self.mae(batch["label"], y_hat)
        self.log_accuracy(batch["label"], y_hat, split)
        self.log(f"{split}_loss", loss, on_step=bool(split=="train"), on_epoch=True, prog_bar=True, logger=True)
        return loss
            
    
    def training_step(self, batch, batch_index):
        return self.step(batch, split="train")
    
    
    def validation_step(self, batch, batch_index):
        self.step(batch, split="val")
        
    
    def test_step(self, batch, batch_index):
        self.step(batch, split="test")
        
    
    def log_accuracy(self, y, y_hat, split):
        """Log accuracy with threshold of 0.5"""
        # acc = self.acc[split]
        acc = getattr(self, f"{split}_acc")
        y = torch.where(y > .5, 1, 0)
        y_hat = torch.where(y_hat > .5, 1, 0)
        acc(y, y_hat)
        self.log(f"{split}_acc", acc, on_step=bool(split=="train"), on_epoch=True, prog_bar=True, logger=True)
    