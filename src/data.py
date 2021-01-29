from typing import NamedTuple

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BertInput(NamedTuple):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor


class TokenizedDataFrameDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 df: pd.DataFrame,
                 x_label: str = 'text',
                 y_label: str = 'label',
                 max_seq_len: int = 20):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.x = df[x_label]
        self.length = len(self.x)

        self.y = df[y_label].astype('category')
        self.n_classes = len(self.y.cat.categories)
        self.y = self.y.cat.codes

    def preprocess_text(self, text: str) -> BertInput:
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_seq_len]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len, f'{len(input_ids)} != {self.max_seq_len}'
        assert len(input_mask) == self.max_seq_len, f'{len(input_mask)} != {self.max_seq_len}'
        assert len(segment_ids) == self.max_seq_len, f'{len(segment_ids)} != {self.max_seq_len}'

        return BertInput(
            input_ids=torch.LongTensor(input_ids),
            input_mask=torch.LongTensor(input_mask),
            segment_ids=torch.LongTensor(segment_ids)
        )

    def preprocess_label(self, label: int):
        result = torch.zeros(self.n_classes).long()
        result[label] = 1
        return result

    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        return {
            'x': self.preprocess_text(x),
            'y': self.preprocess_label(y)
        }

    def __len__(self):
        return self.length

    @staticmethod
    def from_file(tokenizer: BertTokenizer,
                  file_path: str,
                  x_label: str = 'text',
                  y_label: str = 'label',
                  max_seq_len: int = 20):
        df = pd.read_csv(file_path)
        return TokenizedDataFrameDataset(tokenizer, df, x_label, y_label, max_seq_len)
