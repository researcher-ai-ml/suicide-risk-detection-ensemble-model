# models.py
import numpy as np
import torch
torch.manual_seed(42)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import DistilBertTokenizer, DistilBertModel

def get_ml_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            subsample=1.0,
            colsample_bytree=1.0
        )
    }

def build_lstm(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 64),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_distilbert_embeddings(texts, batch_size=8, max_len=64):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(
            list(texts[i:i+batch_size]),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        with torch.no_grad():
            emb = model(**tokens).last_hidden_state.mean(dim=1)
        embeddings.append(emb.numpy())

    return np.vstack(embeddings)
