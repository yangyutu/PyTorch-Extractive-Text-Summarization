import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch

from data_utils.text_data import PretokenizedTextData, build_dataloader
from models.bert_encoder import SentenceEncoder, InterSentenceEncoder
from models.extractive_summarizer import BertExtractiveSummarizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
import pickle
from tqdm import tqdm


def main(args):
    # fix random seeds for reproducibility
    SEED = args.seed
    pl.seed_everything(SEED)
    dataset = PretokenizedTextData(data_dir=args.data_dir, split="test", load_txt=True)
    test_dataloader = build_dataloader(
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

    # wandb_logger = WandbLogger(
    #     project=args.project_name,  #
    #     log_model="all",
    #     save_dir=args.default_root_dir,
    #     group=args.dataset_name,
    #     tags=[args.model_name, args.dataset_name, args.pretrained_model_name],
    # )

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    # disable randomness, dropout, etc...
    model.eval()
    model.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # predict with the model
    scores_all = []
    src_txt_all = []
    tgt_txt_all = []
    src_sent_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = batch.to(device)
            y_hat = model(batch)
            # print(y_hat.shape)
            src_txt_all.extend(batch.src_txt)
            tgt_txt_all.extend(batch.tgt_txt)
            src_sent_labels.extend(batch.src_lent_labels)

            # print(len(src_txt_all), len(tgt_txt_all))
            for i, cls_loc in enumerate(batch.cls_locs):
                scores = y_hat[i, 0 : len(cls_loc), 1].detach().cpu().numpy()
                scores_all.append(scores)
        # print(scores_all)

    output_dict = {
        "src_txt": src_txt_all,
        "tgt_txt": tgt_txt_all,
        "src_sent_labels": src_sent_labels,
        "scores": scores_all,
    }

    with open("infer_all.pkl", "wb") as file:
        pickle.dump(output_dict, file)


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

    parser.add_argument("--truncate", type=int, default=128)
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=str, default=2048)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
