from typing import Any

import torch
from transformers import (
    Adafactor
)
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pytorch_lightning as pl


class InstructFinetuningModel(pl.LightningModule):
    def __init__(self, model='t5-base', *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.drop_out = 0.05

        self.base_model = T5ForConditionalGeneration.from_pretrained(self.model, dropout_rate=self.drop_out)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model)

    def forward(self, batch) -> Any:
        output = self.base_model(
            input_ids = batch['input_ids'],
            labels = batch['labels']
        )

        return output

    def training_step(self, train_batch, batch_idx):
        output = self(train_batch)
        
        return output.loss

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            output = self(val_batch)
            self.log("val_loss", output.loss)
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=self.hparams.learning_rate) #, eps=self.hparams.adam_epsilon

        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

