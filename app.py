import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords

# Load models 
models = {
    "Logistic Regression": joblib.load('./Models/logistic_regression_model.pkl'),
    "Naive Bayes": joblib.load('./Models/naive_bayes_model.pkl'),
    "LSTM": joblib.load('./Models/lstm_model.pkl')
}
vectorizer = joblib.load('./Models/tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction function
def predict_sentiment(model, review_text):
    cleaned_review = clean_text(review_text)
    review_vector = vectorizer.transform([cleaned_review])
    if hasattr(model, "predict_proba"):
        prediction = model.predict(review_vector)
        probability = model.predict_proba(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = probability[0][prediction[0]]
    else:  # For LSTM or custom models
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = None
    return sentiment, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis")
st.markdown("Analyze your movie reviews and find out if they are **positive** or **negative**!")

# Model selection
model_choice = st.selectbox("Choose Model:", list(models.keys()))

st.markdown("---")
review_text = st.text_area("Enter your movie review here:", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if review_text.strip():
        with st.spinner("Analyzing review..."):
            sentiment, confidence = predict_sentiment(models[model_choice], review_text)
        
        # Colored feedback
        if sentiment == "Positive":
            st.success(f"**Predicted Sentiment:** {sentiment}")
        else:
            st.error(f"**Predicted Sentiment:** {sentiment}")
        
        # Confidence bar
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2%}")
            st.progress(int(confidence * 100))
    else:
        st.warning("⚠️ Please enter a review to analyze.")
