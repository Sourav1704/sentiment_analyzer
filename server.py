from flask import Flask, request, jsonify
from flask_cors import CORS

import torch

from transformers import BertTokenizer

from model import BertForSentimentClassification


# =========================
# APP
# =========================

app = Flask(__name__)

CORS(app)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# TOKENIZER
# =========================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# =========================
# MODEL
# =========================

model = BertForSentimentClassification()

model.load_state_dict(
    torch.load(
        "saved_models/sentiment_model.pt",
        map_location=device
    )
)

model.to(device)

model.eval()


# =========================
# ROUTE
# =========================

@app.route("/", methods=["GET"])
def analyze_sentiment():

    text = request.args.get("text")

    if not text:

        return jsonify({
            "error": "No text provided"
        })

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

    sentiment = (
        "Positive"
        if probability > 0.5
        else "Negative"
    )

    percentage = round(
        probability * 100,
        2
    )

    if sentiment == "Negative":

        percentage = round(
            (1 - probability) * 100,
            2
        )

    return jsonify({

        "sentiment": sentiment,

        "percentage": percentage
    })


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5002,
        debug=False
    )

