from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy


class BertExtractiveSummarizer(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        sentence_encoder: nn.Module,
        inter_sentence_encoder: nn.Module,
    ):
        super(BertExtractiveSummarizer, self).__init__()

        self.sentence_encoder = sentence_encoder
        self.inter_sentence_encoder = inter_sentence_encoder
        self.config = config

        # self.train_accuracy = Accuracy(num_classes=num_classes)
        # self.val_accuracy = Accuracy(num_classes=num_classes)

    def forward(self, batch):

        cls_vecs, cls_mask = self.sentence_encoder(batch)
        sent_scores = self.inter_sentence_encoder(cls_vecs, cls_mask)

        return sent_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config["lr"])
        return optimizer

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)
        logits = logits.view(-1, logits.shape[-1])
        targets = batch.cls_labels_padded.view(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=batch.cls_labels_pad_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=batch.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # self.train_accuracy.update(logits, target)
        return loss

    def validation_step(self, batch, batch_idx):

        logits = self.forward(batch)
        logits = logits.view(-1, logits.shape[-1])
        targets = batch.cls_labels_padded.view(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=batch.cls_labels_pad_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=batch.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # self.val_accuracy.update(logits, target)

        return loss
