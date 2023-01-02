import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data_utils.text_data import PretokenizedTextData, build_dataloader
from models.bert_encoder import SentenceEncoder, InterSentenceEncoder
from models.extractive_summarizer import BertExtractiveSummarizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger


def main(args):
    # fix random seeds for reproducibility
    SEED = args.seed
    pl.seed_everything(SEED)
    dataset = PretokenizedTextData(data_dir=args.data_dir, split="train")
    train_dataloader = build_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    dataset = PretokenizedTextData(data_dir=args.data_dir, split="valid")
    val_dataloader = build_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    config = {}
    config["lr"] = args.lr

    encoder = SentenceEncoder(args.pretrained_model_name)
    inter_sent_encoder = InterSentenceEncoder(
        d_model=args.d_model,
        d_ff=args.d_ff,
        heads=args.num_heads,
        num_inter_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = BertExtractiveSummarizer(
        config=config,
        sentence_encoder=encoder,
        inter_sentence_encoder=inter_sent_encoder,
    )

    wandb_logger = WandbLogger(
        project=args.project_name,  #
        log_model="all",
        save_dir=args.default_root_dir,
        group=args.dataset_name,
        tags=[args.model_name, args.dataset_name, args.pretrained_model_name],
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        #    precision=args.precision,
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            lr_monitor,
        ],
        # gradient_clip_val=config.trainer.max_grad_norm,
        deterministic=True,  # RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
    )

    # 4. Train!
    trainer.fit(model, train_dataloader, val_dataloader)


def parse_arguments():

    parser = argparse.ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)

    # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--truncate", type=int, default=128)
    parser.add_argument("--pretrained_model_name", type=str, default="")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=str, default=2048)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
