Sentiment Analysis Using NLP (End-to-End ML Project)

This project builds a Natural Language Processing (NLP) model to classify customer reviews into Positive, Negative, or Neutral sentiments.
The final model is deployed using Streamlit for real-time sentiment prediction.

📖 Project Overview

The objective of this project is to analyze product reviews and automatically classify their sentiment.
The dataset contains 1,440 customer reviews.

The project covers the complete machine learning pipeline:

Data Cleaning

Exploratory Data Analysis (EDA)

Text Preprocessing

Feature Engineering

Model Training

Model Evaluation

Deployment

🧹 Text Preprocessing

Lowercasing text

Removing punctuation and special characters

Removing stopwords using NLTK

Text normalization

📊 Feature Engineering

TF-IDF Vectorization

🤖 Machine Learning Models Used

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

📈 Model Evaluation

The models were evaluated using:

Accuracy

Precision

Recall

F1-Score

Best Model Accuracy: 87%

The SVM model achieved the best performance with balanced results across all three sentiment classes.

🌐 Deployment

The final model is deployed using Streamlit.

Features of the Web App:

Real-time sentiment prediction

Word and character count

Prediction history

Run locally using:

streamlit run app.py

🛠️ Technologies & Libraries

Python

Pandas

NumPy

NLTK

Scikit-learn

Matplotlib

Seaborn

WordCloud

Streamlit

📂 Project Structure
├── Sentiment_Analysis.ipynb   # Model development
├── app.py                     # Streamlit application
├── dataset.csv                # Dataset
├── README.md                  # Project documentation
