import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import BertTokenizer

from dataset import SSTDataset
from model import BertForSentimentClassification
from utils import get_accuracy_from_logits


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)


# =========================
# TOKENIZER
# =========================

print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("Tokenizer loaded!")


# =========================
# DATASETS
# =========================

print("\nLoading datasets...")

train_dataset = SSTDataset(
    filename="../data/train.tsv",
    tokenizer=tokenizer,
    maxlen=128
)

dev_dataset = SSTDataset(
    filename="../data/dev.tsv",
    tokenizer=tokenizer,
    maxlen=128
)

print("Datasets loaded!")


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=1
)

print("Dataloaders ready!")


# =========================
# MODEL
# =========================

print("\nLoading model...")

model = BertForSentimentClassification()

print("Model loaded!")

model.to(device)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

print("Optimizer ready!")


# =========================
# TRAINING
# =========================

epochs = 1

print("\nStarting training...")

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_acc = 0

    for i, batch in enumerate(train_loader):

        print(f"\nProcessing batch {i}")

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = logits.squeeze(-1)

        loss = criterion(
            logits,
            labels
        )

        acc = get_accuracy_from_logits(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        total_acc += acc.item()

    avg_loss = total_loss / len(train_loader)

    avg_acc = total_acc / len(train_loader)

    print(
        f"\nEpoch {epoch + 1}"
        f" | Loss: {avg_loss:.4f}"
        f" | Accuracy: {avg_acc:.4f}"
    )


# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "saved_models/sentiment_model.pt"
)

print("\nModel saved successfully!")


# =========================
# INTERACTIVE TESTING
# =========================

model.eval()

print("\n==========================")
print("Sentiment Analyzer Ready")
print("==========================")

while True:

    text = input(
        "\nEnter text (type quit to stop): "
    )

    if text.lower() == "quit":

        print("\nExiting...")

        break

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding[
        "input_ids"
    ].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probability = torch.sigmoid(
            logits
        ).item()

    print("\nProbability:", probability)

    if probability > 0.5:

        print("Positive Sentiment")

    else:

        print("Negative Sentiment")
