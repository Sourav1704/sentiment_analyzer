import torch
import pandas as pd

from torch.utils.data import Dataset


class SSTDataset(Dataset):

    def __init__(self, filename, tokenizer, maxlen):

        self.data = pd.read_csv(
            filename,
            sep="\t"
        )

        # LIMIT DATASET SIZE
        self.data = self.data.head(1000)

        self.tokenizer = tokenizer

        self.maxlen = maxlen

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        sentence = str(
            self.data.iloc[index]["sentence"]
        )

        label = float(
            self.data.iloc[index]["label"]
        )

        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.maxlen,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding[
            "attention_mask"
        ].squeeze(0)

        return {

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "label": torch.tensor(
                label,
                dtype=torch.float
            )
        }