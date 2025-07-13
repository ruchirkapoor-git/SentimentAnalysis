# sentiment_analysis_app.py

import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import pickle
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Data Collection & Preprocessing
@st.cache_data
def load_and_train():
    # Load Sentiment140 dataset
    df = pd.read_csv(
        'C:/Users/Ruchir Kapoor/OneDrive/Desktop/Data/training.1600000.processed.noemoticon.csv',
        encoding='latin-1',
        header=None
    )
    df = df[[0, 5]]
    df.columns = ['polarity', 'text']

    # Keep only positive (4) and negative (0) sentiments
    df = df[df['polarity'] != 2]
    df['polarity'] = df['polarity'].map({0: 0, 4: 1})

    # Text Preprocessing
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    # Feature Engineering
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['polarity']

    # Model Building
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Return for immediate use
    return model, vectorizer

# 2. Load model and vectorizer if they exist, else train and save
if os.path.exists('sentiment_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
else:
    model, vectorizer = load_and_train()

# 3. Text Preprocessing (for new input)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# 4. Streamlit App
st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter a comment or tweet to analyze its sentiment:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    st.write(f"Sentiment: **{sentiment}**")
