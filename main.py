from preprocess import load_and_preprocess
from model import train_model

if _name_ == "_main_":
    print("📥 Loading and preprocessing data...")
    train_texts, val_texts, train_labels, val_labels = load_and_preprocess("amazon_multilingual.csv")

    print("🚀 Training model with mBERT...")
    train_model(train_texts, train_labels, val_texts, val_labels)

    print("✅ Training complete. Model saved to 'mbert-sentiment-model/'")