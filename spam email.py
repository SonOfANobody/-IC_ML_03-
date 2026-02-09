import streamlit as st
import joblib
import pandas as pd
from textblob import TextBlob
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tesla/Apple Spam Grader", page_icon="🚀")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    model = joblib.load('spam_classification_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    return model, tfidf, scaler

try:
    model, tfidf, scaler = load_models()
except Exception as e:
    st.error("Error loading model files. Ensure .pkl files are in the same directory.")

# --- APP INTERFACE ---
st.title("🛡️ Smart Spam Classifier")
st.markdown("""
This model uses **LightGBM** and **NLP** to grade messages. 
Optimized for deployment with a footprint under 25MB.
""")

message = st.text_area("Enter message to analyze:", placeholder="e.g., 'You won a free Tesla! Click here to claim.'")

if st.button("Analyze & Grade"):
    if message.strip():
        # 1. Feature Engineering (Same as training)
        clean_text = message.lower().replace(r'[^a-z\s]', '')
        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        avg_word_len = len(clean_text.split()) if len(clean_text.split()) > 0 else 0
        
        # 2. Transform
        text_vec = tfidf.transform([clean_text])
        num_features = scaler.transform([[polarity, subjectivity, avg_word_len]])
        
        # 3. Combine & Predict
        from scipy.sparse import hstack
        combined = hstack([text_vec, num_features])
        prediction = model.predict(combined)[0]
        
        # 4. Results
        st.divider()
        st.subheader(f"Result: {prediction}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity", f"{polarity:.2f}")
        col2.metric("Subjectivity", f"{subjectivity:.2f}")
        col3.metric("Word Count", avg_word_len)
    else:
        st.warning("Please enter some text first!")