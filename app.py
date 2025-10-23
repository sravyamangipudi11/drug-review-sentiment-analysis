# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import string
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Functions
# -----------------------------
def review_clean(review):
    """Clean review text."""
    lower = review.lower()
    lower = lower.replace("&#039;", "")
    lower = ''.join([c if c.isascii() else ' ' for c in lower])
    lower = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in lower])
    lower = ' '.join(lower.split())
    return lower

# Feature engineering for a single review input
def create_features(df):
    df['sentiment'] = df['review_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['count_word'] = df['review_clean'].apply(lambda x: len(str(x).split()))
    df['count_unique_word'] = df['review_clean'].apply(lambda x: len(set(str(x).split())))
    df['count_letters'] = df['review_clean'].apply(lambda x: len(str(x)))
    df['count_punctuations'] = df['review'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['count_words_upper'] = df['review'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df['count_words_title'] = df['review'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df['count_stopwords'] = df['review'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    df['mean_word_len'] = df['review_clean'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    return df

# -----------------------------
# Load Model & Encoders
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("lgbm_model.pkl")
    le_drug = joblib.load("label_encoder_drug.pkl")
    le_condition = joblib.load("label_encoder_condition.pkl")
    return model, le_drug, le_condition

model, le_drug, le_condition = load_model()

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Drug Review Sentiment Predictor", layout="wide")
st.title("💊 Drug Review Sentiment Predictor")

# Dropdowns
drug_names = le_drug.classes_
conditions = le_condition.classes_

st.header("Enter Review Details")
drug_input = st.selectbox("Drug Name", drug_names)
condition_input = st.selectbox("Condition", conditions)
review_input = st.text_area("Review Text")
useful_count_input = st.number_input("Useful Count", min_value=0, value=0)
day_input = st.number_input("Day of Review", min_value=1, max_value=31, value=1)
month_input = st.number_input("Month of Review", min_value=1, max_value=12, value=1)
year_input = st.number_input("Year of Review", min_value=2000, max_value=2030, value=2024)

# Predict Button
if st.button("Predict Sentiment"):
    # Clean review
    review_cleaned = review_clean(review_input)
    
    # Create mini dataframe
    df = pd.DataFrame({
        'review': [review_input],
        'review_clean': [review_cleaned],
        'usefulCount': [useful_count_input],
        'day': [day_input],
        'month': [month_input],
        'Year': [year_input]
    })
    
    # Feature engineering
    df = create_features(df)
    
    # Encode categorical
    df['drugName_LE'] = le_drug.transform([drug_input])
    df['condition_LE'] = le_condition.transform([condition_input])
    
    # Placeholder freq and mean sentiment
    df['drugName_freq'] = 0
    df['drugName_mean_sentiment'] = 0
    df['condition_freq'] = 0
    df['condition_mean_sentiment'] = 0
    df['sentiment_clean_ss'] = df['sentiment']
    
    # Feature order
    feature_cols = [
        'usefulCount', 'sentiment', 'day', 'month', 'Year', 'sentiment_clean_ss', 
        'count_word', 'count_unique_word', 'count_letters', 'count_punctuations',
        'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len',
        'drugName_LE', 'drugName_freq', 'drugName_mean_sentiment',
        'condition_LE', 'condition_freq', 'condition_mean_sentiment'
    ]
    
    X = df[feature_cols]
    
    # Prediction
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)
    
    sentiment_label = "Positive" if pred[0] == 1 else "Negative"
    st.success(f"Predicted Sentiment: {sentiment_label}")
    st.info(f"Confidence: {pred_prob[0][pred[0]]*100:.2f}%")
