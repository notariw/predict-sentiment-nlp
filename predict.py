from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model dan tokenizer dari folder lokal
MODEL_PATH = "mbert-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

labels = ["negative", "neutral", "positive"]  # Ubah sesuai label training kamu

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs).item()
    return labels[predicted_class], probs[0][predicted_class].item()
