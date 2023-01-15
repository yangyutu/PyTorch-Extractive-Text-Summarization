import torch
from torch.utils.data import Dataset, DataLoader
import os, glob, bisect


def collate_tokens(values, pad_idx, left_pad=False, pad_to_length=None):
    # Simplified version of `collate_tokens` from fairseq.data.data_utils
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


class PretokenizedTextData(Dataset):
    def __init__(self, data_dir, split="train", max_pos=512, load_txt=False):
        super().__init__()

        self.raw_data = []
        filenames = glob.glob(data_dir + "/[a-z]*" + split + ".[0-9]*.pt")
        print(f"loading {len(filenames)} files in {split} partition")
        for filename in filenames:
            data = torch.load(filename)
            self.raw_data.extend(data)
        self.max_pos = max_pos
        self.load_txt = load_txt

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.preprocess(self.raw_data[index])

    def preprocess(self, example):
        src_token_ids = example["src"]
        src_sent_labels = example["src_sent_labels"]
        cls_locs = example["clss"]

        end_id = [src_token_ids[-1]]
        src_token_ids = src_token_ids[:-1][: self.max_pos - 1] + end_id
        max_sent_id = bisect.bisect_left(cls_locs, self.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        cls_locs = cls_locs[:max_sent_id]

        out = {
            "src_token_ids": src_token_ids,
            "src_len": len(src_token_ids),
            "cls_locs": cls_locs,
            "cls_len": len(cls_locs),
            "src_sent_labels": src_sent_labels,
        }

        if self.load_txt:
            out.update({"src_txt": example["src_txt"], "tgt_txt": example["tgt_txt"]})

        return out


class Batch:
    def __init__(self, data, pad_id=0):
        # a list of dictionary to a dictionary of lists
        data = {key: [i[key] for i in data] for key in data[0]}
        src_token_ids, src_len, cls_locs, src_sent_labels = (
            data["src_token_ids"],
            data["src_len"],
            data["cls_locs"],
            data["src_sent_labels"],
        )
        self.pad_id = pad_id
        self.batch_size = len(src_token_ids)
        # Encoder info
        self.token_input, self.token_attn_mask = None, None
        self.cls_input, self.cls_attn_mask = None, None
        self.cls_labels_pad_idx = -1
        self.src_sent_labels = src_sent_labels
        # Build batch inputs
        self.init_encoder_token_seq(src_token_ids, src_len, cls_locs, src_sent_labels)

        if "src_txt" in data and "tgt_txt" in data:
            self.src_txt = data["src_txt"]
            self.tgt_txt = data["tgt_txt"]

    def init_encoder_token_seq(self, src, src_len, cls_locs, src_sent_labels):

        self.token_input = collate_tokens(values=src, pad_idx=self.pad_id)
        self.token_len = torch.LongTensor(src_len)
        self.token_attn_mask = self.token_input != self.pad_id

        self.cls_locs = cls_locs
        self.cls_locs_padded = collate_tokens(values=cls_locs, pad_idx=-1)
        self.cls_labels_padded = collate_tokens(
            values=src_sent_labels, pad_idx=self.cls_labels_pad_idx
        )
        self.cls_attn_mask = self.cls_locs_padded != -1
        self.cls_locs_padded[self.cls_locs_padded == -1] = 0

    def __len__(self):
        return self.token_input.size(0)

    def __str__(self):
        batch_info = {
            "token_input": self.token_input,  # [B x L]
            "token_len": self.token_len,  # [B]
            "token_attn_mask": self.token_attn_mask,
            "cls_locs": self.cls_locs,
            "cls_locs_len": [len(cls_loc) for cls_loc in self.cls_locs],
            "cls_locs_padded": self.cls_locs_padded,
            "cls_attn_mask": self.cls_attn_mask,
            "cls_labels_padded": self.cls_labels_padded,
            "cls_labels_pad_idx": self.cls_labels_pad_idx,
        }
        return str(batch_info)

    def to(self, device):
        self.token_input = self.token_input.to(device)
        self.token_attn_mask = self.token_attn_mask.to(device)
        self.cls_attn_mask = self.cls_attn_mask.to(device)
        self.cls_locs_padded = self.cls_locs_padded.to(device)
        self.cls_labels_padded = self.cls_labels_padded.to(device)

        return self


def build_dataloader(dataset, pad_idx=0, batch_size=32, num_workers=8, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda data, pad_idx=pad_idx: Batch(data=data, pad_id=pad_idx),
        num_workers=num_workers,
    )
    return data_loader


def _test_data():
    data_dir = "/mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_ext"
    dataset = PretokenizedTextData(data_dir=data_dir)
    print(dataset[0])


def _test_data_loader():
    data_dir = "/mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_ext"
    dataset = PretokenizedTextData(data_dir=data_dir)
    data_loader = build_dataloader(dataset, batch_size=2)

    for batch in data_loader:
        print(batch)
        break


if __name__ == "__main__":
    _test_data_loader()
