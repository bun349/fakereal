import streamlit as st
from newspaper import Article
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan tokenizer
model = tf.keras.models.load_model("lstm_fake_news_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# Konfigurasi
max_len = 200

def preprocess_text(text):
    # Lowercase dan hapus karakter khusus jika mau
    return text.lower()

def predict_news(text):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_len)
    prob = model.predict(padded)[0][0]
    label = "REAL" if prob >= 0.5 else "FAKE"
    return label, prob

def fetch_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

# UI Streamlit
st.title("📰 Fake News Detector (LSTM Model)")

url_input = st.text_input("Masukkan URL berita")

if url_input:
    with st.spinner("Mengambil artikel..."):
        article_text = fetch_article_text(url_input)

    if article_text:
        st.subheader("📄 Isi Berita (Rangkuman)")
        st.write(article_text[:1000] + " ...")

        label, prob = predict_news(article_text)
        st.subheader("🔍 Hasil Deteksi:")
        st.write(f"**Prediksi:** {label}")
        st.write(f"**Probabilitas:** {prob:.2f}")
    else:
        st.error("Gagal mengambil atau memproses artikel dari URL.")
