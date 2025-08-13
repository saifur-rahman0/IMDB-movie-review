import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords

# Load models and vectorizer
models = {
    "Logistic Regression": joblib.load('./Models/logistic_regression_model.pkl'),
    "Naive Bayes": joblib.load('./Models/naive_bayes_model.pkl'),
    "LSTM": joblib.load('./Models/lstm_model.pkl')
}
vectorizer = joblib.load('./Models/tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_sentiment(model, review_text):
    cleaned_review = clean_text(review_text)
    review_vector = vectorizer.transform([cleaned_review])
    if hasattr(model, "predict_proba"):
        prediction = model.predict(review_vector)
        probability = model.predict_proba(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = probability[0][prediction[0]]
    else:  # For LSTM (if using scikit-learn wrapper)
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = None
    return sentiment, confidence

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Select a model, enter your review, and get the sentiment prediction.")

model_choice = st.selectbox("Choose Model", list(models.keys()))

st.markdown("<br><br>", unsafe_allow_html=True)
review_text = st.text_area("**Enter your review:**")

if st.button("Predict"):
    if review_text.strip():
        sentiment, confidence = predict_sentiment(models[model_choice], review_text)
        st.write(f"**Predicted Sentiment:** {sentiment}")
        if confidence is not None:
            st.write(f"**Confidence/Precision:** {confidence:.2%}")
    else:
        st.write("Please enter a review.")
