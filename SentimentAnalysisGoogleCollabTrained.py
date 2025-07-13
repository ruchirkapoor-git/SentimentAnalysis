import pandas as pd
import re
import nltk
import streamlit as st
import pickle
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load pre-trained model and vectorizer (no retraining)
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 2. Text Preprocessing (for new input)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# 3. Streamlit App
st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter a comment or tweet to analyze its sentiment:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    st.write(f"Sentiment: **{sentiment}**")
