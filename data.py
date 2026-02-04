# data.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def load_data(csv_path, sample_frac=0.05):
    df = pd.read_csv(csv_path)
    df = df[['text', 'class']].dropna()
    df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0}).astype(int)

    df = df.sample(frac=sample_frac, random_state=42)
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text'].values
    y = df['label'].values

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
