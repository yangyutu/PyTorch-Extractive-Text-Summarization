from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_utils.eval_rouge import eval_rouge
import numpy as np


class BertExtractiveSummarizer(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        sentence_encoder: nn.Module,
        inter_sentence_encoder: nn.Module,
    ):
        super(BertExtractiveSummarizer, self).__init__()
        self.save_hyperparameters()
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
        lr_warmup_steps = self.config["lr_warm_up_steps"]

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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
        scores = logits[:, :, 1]
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

        # compute rouge scores for this batch
        scores_valid_all = []
        for i, cls_loc in enumerate(batch.cls_locs):
            scores_valid = scores[i, 0 : len(cls_loc)].detach().cpu().numpy()
            scores_valid_all.append(scores_valid)

        rouge_score_list = eval_rouge(
            batch.src_txt, batch.tgt_txt, batch.src_sent_labels, scores_valid_all
        )

        return {"val_loss": loss, "rouge_score_list": rouge_score_list}

    def validation_epoch_end(self, testoutputs):

        rouge_1_f_all, rouge_2_f_all, rouge_l_f_all = [], [], []
        for output in testoutputs:
            for score in output["rouge_score_list"]:
                rouge_1_f_all.append(score["rouge-1"]["f"])
                rouge_2_f_all.append(score["rouge-2"]["f"])
                rouge_l_f_all.append(score["rouge-l"]["f"])

        rouge_1_f = np.mean(rouge_1_f_all)
        rouge_2_f = np.mean(rouge_2_f_all)
        rouge_l_f = np.mean(rouge_l_f_all)

        self.log("rouge-1-f-epoch", rouge_1_f, prog_bar=True, logger=True)
        self.log("rouge-2-f-epoch", rouge_2_f, prog_bar=True, logger=True)
        self.log("rouge-l-f-epoch", rouge_l_f, prog_bar=True, logger=True)
