import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        # mask convention: 0 for valid position, 1 for invalid position
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)

        out = self.feed_forward(self.dropout(context) + inputs)

        return out


class SentenceEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, batch):
        encoded_inputs = {
            "input_ids": batch.token_input,
            "attention_mask": batch.token_attn_mask,
        }
        encoder_outputs = self.encoder(**encoded_inputs)
        # encoder_outputs.last_hidden_state: [batch_size x seq_len x d_model]
        cls_vecs = encoder_outputs.last_hidden_state[
            torch.arange(batch.token_input.size(0)).unsqueeze(1), batch.cls_locs_padded
        ]
        return cls_vecs, batch.cls_attn_mask


class InterSentenceEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super().__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pred = nn.Linear(d_model, 2, bias=True)

    def forward(self, cls_vecs, cls_mask):
        """See :obj:`EncoderBase.forward()`"""
        # cls_mask: 1 for non-padded position, 0 for padded position
        batch_size, n_sents = cls_vecs.size(0), cls_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]  # pos_emb: [1 x cls_seq_len x d_model]
        x = (
            cls_vecs * cls_mask[:, :, None].float()
        )  # x: [batch_size, cls_seq_len x d_model]
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](
                i, x, x, ~cls_mask
            )  # all_sents * max_tokens * dim

        x = self.layer_norm(x)  # x shape: batch_size x seq_len x embed_dim
        sent_scores = self.pred(x)

        return sent_scores


class HGInterSentenceEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)

        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.update(
            {
                "num_attention_heads": heads,
                "hidden_size": d_model,
                "intermediate_size": d_ff,
                "num_hidden_layers": num_inter_layers,
            }
        )
        self.transformer_inter = BertEncoder(config)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pred = nn.Linear(d_model, 2, bias=True)

    def forward(self, cls_vecs, cls_mask):
        """See :obj:`EncoderBase.forward()`"""
        # cls_mask: 1 for non-padded position, 0 for padded position
        batch_size, n_sents = cls_vecs.size(0), cls_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]  # pos_emb: [1 x cls_seq_len x d_model]
        x = (
            cls_vecs * cls_mask[:, :, None].float()
        )  # x: [batch_size, cls_seq_len x d_model]
        x = self.layer_norm(x + pos_emb)

        mask = (
            cls_mask.float()
            .masked_fill(cls_mask == 0, float("-inf"))
            .masked_fill(cls_mask == 1, float(0.0))
        )
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L994
        mask = mask[:, None, None, :]
        embedding = self.transformer_inter(x, attention_mask=mask)
        encoded_embeddings = embedding.last_hidden_state
        # encoded_embeddings shape: batch_size x seq_len x embed_dim
        sent_scores = self.pred(encoded_embeddings)

        return sent_scores


def _test_token_encoder():
    from data_utils.text_data import PretokenizedTextData, build_dataloader

    data_dir = "/mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_ext"
    dataset = PretokenizedTextData(data_dir=data_dir)
    data_loader = build_dataloader(dataset, batch_size=6)
    encoder = SentenceEncoder("bert-base-uncased")
    inter_sent_encoder = InterSentenceEncoder(
        d_model=768, d_ff=2048, heads=8, num_inter_layers=2, dropout=0.2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in data_loader:
        batch.to(device)
        encoder.to(device)
        inter_sent_encoder.to(device)
        cls_vecs, cls_mask = encoder(batch)
        sent_scores = inter_sent_encoder(cls_vecs, cls_mask)
        print(sent_scores)
        break


def _test_token_hg_encoder():
    from data_utils.text_data import PretokenizedTextData, build_dataloader

    data_dir = "/mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_ext"
    dataset = PretokenizedTextData(data_dir=data_dir, split="valid")
    data_loader = build_dataloader(dataset, batch_size=6)
    encoder = SentenceEncoder("bert-base-uncased")
    inter_sent_encoder = HGInterSentenceEncoder(
        d_model=768, d_ff=2048, heads=8, num_inter_layers=2, dropout=0.2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in data_loader:
        batch.to(device)
        encoder.to(device)
        inter_sent_encoder.to(device)
        cls_vecs, cls_mask = encoder(batch)
        sent_scores = inter_sent_encoder(cls_vecs, cls_mask)
        print(sent_scores.shape)
        break


if __name__ == "__main__":
    # _test_token_encoder()
    _test_token_hg_encoder()
