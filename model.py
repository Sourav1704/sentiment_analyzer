import torch
import torch.nn as nn

from transformers import BertModel


class BertForSentimentClassification(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 1)
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        return logits