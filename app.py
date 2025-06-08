import streamlit as st
from predict import predict_sentiment

st.set_page_config(page_title="Prediksi Sentimen Multilingual", layout="centered")

st.title("📊 Prediksi Sentimen Review")
st.write("Masukkan teks ulasan/review dalam bahasa apapun:")

user_input = st.text_area("Masukkan teks:", height=150)

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        label, confidence = predict_sentiment(user_input)
        st.success(f"**Sentimen: {label.upper()}** (Confidence: {confidence:.2f})")
