from typing import NamedTuple

import torch
import pandas as pd
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class BertInput(NamedTuple):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor


class TokenizedDataFrameDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 file_path: str,
                 x_label: str = 'text',
                 y_label: str = 'label',
                 max_seq_len: int = 20):
        """
        :param data_path: path to data
        """
        self.tokenizer = tokenizer
        self.x_label = x_label
        self.y_label = y_label
        self.max_seq_len = max_seq_len

        self.df = pd.read_csv(file_path)
        self.df[y_label] = self.df[y_label].astype('category')

        self.y_labels = self.df[y_label].cat.categories
        self.df[y_label] = self.df[y_label].cat.codes

    def preprocess_text(self, text: str) -> BertInput:
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        return BertInput(*[torch.LongTensor(_) for _ in [input_ids, input_mask, segment_ids]])

    def preprocess_label(self, label: int):
        result = torch.zeros(len(self.y_labels))
        result[label] = 1
        return result

    def __getitem__(self, index) -> dict:
        sample = self.df.iloc[index]
        x = sample[self.x_label]
        y = sample[self.y_label]
        return {
            'x': self.preprocess_text(x),
            'y': self.preprocess_label(y)
        }

    def __len__(self):
        return len(self.df)
