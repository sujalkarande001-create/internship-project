# -*- coding: utf-8 -*-

!pip install -q scikit-learn pandas numpy

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------- POSITIVE SENTENCES ----------------
positive = [
    "I absolutely love this product",
    "This product is excellent",
    "Amazing quality and great performance",
    "Very happy with this purchase",
    "Fantastic customer service experience",
    "Extremely satisfied with the product",
    "Highly recommended for everyone",
    "Perfect experience overall",
    "Great value for money",
    "Outstanding performance and quality",
    "The service exceeded my expectations",
    "I am very pleased with the results",
    "Superb experience using this product",
    "This product works perfectly",
    "Excellent design and build quality",
    "The app runs smoothly and fast",
    "Very impressive performance",
    "The quality is top notch",
    "I am completely satisfied",
    "Best product I have ever used"
]

# ---------------- NEGATIVE SENTENCES ----------------
negative = [
    "I hate this product",
    "This product is terrible",
    "Worst experience ever",
    "Very disappointed with the quality",
    "Poor performance and bad service",
    "Extremely unhappy with this purchase",
    "Waste of money",
    "The product stopped working quickly",
    "Terrible customer service experience",
    "Not satisfied at all",
    "The quality is very poor",
    "This is the worst product ever",
    "Highly disappointing experience",
    "The app crashes frequently",
    "Bad design and low quality",
    "The service was horrible",
    "Very frustrating experience",
    "Totally useless product",
    "I regret buying this product",
    "The performance is extremely poor"
]

# ---------------- NEUTRAL SENTENCES ----------------
neutral = [
    "The product is average",
    "This product is okay",
    "Service was acceptable",
    "Neither good nor bad experience",
    "The quality is normal",
    "Product works as expected",
    "Average performance overall",
    "Nothing special about this product",
    "The experience was normal",
    "Fair enough for the price",
    "The service was fine",
    "Just an ordinary product",
    "Meets basic expectations",
    "The results were average",
    "Standard quality product",
    "Not too good not too bad",
    "The app works fine",
    "Typical experience",
    "Acceptable performance",
    "Overall okay experience"
]

tweets = []
labels = []

for i in range(30):   # 20 × 30 = 600 per class
    for t in positive:
        tweets.append(f"{t} #{i}")
        labels.append("Positive")
    for t in negative:
        tweets.append(f"{t} #{i}")
        labels.append("Negative")
    for t in neutral:
        tweets.append(f"{t} #{i}")
        labels.append("Neutral")

df = pd.DataFrame({
    "Tweet": tweets,
    "Sentiment": labels
})

df.to_csv("twitter_ml_dataset_1800.csv", index=False)

print("Total Tweets:", len(df))
print(df["Sentiment"].value_counts())

df = pd.read_csv("twitter_ml_dataset_1800.csv")
df.head()

X = df["Tweet"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

test_sentences = [
    "I absolutely good this product",
    "This product is terrible",
    "The service was acceptable",
    "Worst quality and horrible experience",
    "Amazing performance and great quality"
]

test_vec = vectorizer.transform(test_sentences)
predictions = model.predict(test_vec)

for t, p in zip(test_sentences, predictions):
    print(f"Tweet: {t}")
    print(f"Predicted Sentiment: {p}\n")

