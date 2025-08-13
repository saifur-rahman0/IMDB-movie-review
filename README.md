# IMDB Movie Review Sentiment Analysis
#
## Dataset
Source: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset)

## Project Overview
This project performs sentiment analysis on IMDB movie reviews using three models: Logistic Regression, Naive Bayes, and a Deep Learning LSTM. The workflow includes data cleaning, feature engineering (TF-IDF and tokenization), model training, evaluation, and deployment.

## Approach & Tools
- **Data Cleaning:** HTML tags, punctuation, numbers removed; text lowercased; stopwords removed (NLTK).
- **Feature Engineering:** TF-IDF vectorization for ML models; Keras Tokenizer and padding for LSTM.
- **Models:**
	- Logistic Regression (scikit-learn)
	- Naive Bayes (scikit-learn)
	- LSTM (Keras/TensorFlow)
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- **Deployment:** Streamlit app (`app.py`) allows users to select a model, input a review, and get sentiment prediction with confidence.

## Results

### Logistic Regression
Accuracy: 0.8878
Precision: 0.8830
Recall: 0.8940
F1-score: 0.8885

### Naive Bayes
Accuracy: 0.8606
Precision: 0.8742
Recall: 0.8424
F1-score: 0.8580

### LSTM (Bonus)
Accuracy: 0.88
Precision: 0.87
Recall: 0.89
F1-score: 0.88

## Demo
Run `app.py` with Streamlit to interactively test sentiment prediction. Choose a model, enter a review, and view the predicted sentiment and confidence.
