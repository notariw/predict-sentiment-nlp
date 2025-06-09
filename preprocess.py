import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def label_sentiment(star):
    if star <= 2:
        return 0  # negative
    elif star == 3:
        return 1  # neutral
    else:
        return 2  # positive

def load_and_preprocess(max_samples=10000):
    # Gabungkan semua CSV
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/validation.csv")
    test_df = pd.read_csv("data/test.csv")

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df = df[['review_body', 'stars', 'language']].dropna()

    # Bersihkan dan labeli
    df['cleaned_text'] = df['review_body'].astype(str).apply(clean_text)
    df['label'] = df['stars'].apply(label_sentiment)

    # Ambil sample jika lebih dari batas
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Split train:val:test = 70:15:15
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    return (
        train_texts.tolist(), train_labels.tolist(),
        val_texts.tolist(), val_labels.tolist(),
        test_texts.tolist(), test_labels.tolist()
    )
