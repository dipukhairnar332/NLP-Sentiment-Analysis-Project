# NLP-Sentiment-Analysis-Project

A Natural Language Processing (NLP) project for sentiment analysis that classifies text into Positive, Negative, or Neutral. Deployed using Streamlit for real-time sentiment prediction

Overview
This repository contains a Natural Language Processing (NLP) project focused on analyzing and classifying the sentiment of customer product reviews. The goal is to build a robust model capable of classifying text into Positive, Negative, or Neutral sentiment categories, followed by an interactive Streamlit web application for real-time analysis.

The analysis is documented in the Jupyter Notebook: Sentiment_Analysis (1).ipynb.

Deployment and Interaction: Handled by the Python application (app.py), which uses Streamlit to provide an interactive web interface for real-time sentiment prediction.

🚀 Features
Data Acquisition and Cleaning: Loads and processes a dataset of 1440 product reviews.

Exploratory Data Analysis (EDA): Includes analysis of data structure, check for missing values and duplicates (none found).

Sentiment Mapping: Converts numerical ratings (1-5) into categorical sentiment labels (1-2 = Negative, 3 = Neutral, 4-5 = Positive).

Text Preprocessing: Cleans review text by performing lowercasing, removing punctuation/special characters, and eliminating common English stopwords.

Frequent Word Analysis: Identifies the top 10 most frequent words for each sentiment class (Positive, Negative, Neutral).

Interactive Web App: A Streamlit application is included for users to input custom text, get an instant sentiment prediction, view associated text statistics (word/character count), and maintain a prediction history.

🛠️ Technologies & Libraries
The project is built entirely in Python and uses the following key libraries:

Data Manipulation & Analysis: pandas, numpy

Natural Language Processing (NLP): nltk (for stopwords), re

Visualization: matplotlib.pyplot, seaborn, wordcloud

Interactive Application: streamlit (implied by app code)

