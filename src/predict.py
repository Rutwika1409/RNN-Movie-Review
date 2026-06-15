"""
predict.py

This module:
1. Loads the trained RNN model.
2. Preprocesses user reviews.
3. Predicts sentiment.
4. Returns confidence score.
"""

import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# Configuration
# =====================================================

VOCAB_SIZE = 10000

MAX_LENGTH = 200


# =====================================================
# Load IMDB Word Index
# =====================================================

word_index = imdb.get_word_index()


# =====================================================
# Load Model
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "imdb_rnn.keras"
)

model = load_model(MODEL_PATH)


# =====================================================
# Convert Text To Sequence
# =====================================================

def text_to_sequence(review):
    """
    Convert review text into IMDB word indices.
    """

    words = review.lower().split()

    sequence = []

    for word in words:

        if word in word_index:

            index = word_index[word] + 3

            if index < VOCAB_SIZE:

                sequence.append(index)

        else:

            sequence.append(2)

    return sequence


# =====================================================
# Preprocess Review
# =====================================================

def preprocess_review(review):
    """
    Convert review into padded sequence.
    """

    sequence = text_to_sequence(
        review
    )

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH
    )

    return padded_sequence


# =====================================================
# Predict Sentiment
# =====================================================

def predict_sentiment(review):
    """
    Predict review sentiment.
    """

    processed_review = preprocess_review(
        review
    )

    prediction = model.predict(
        processed_review,
        verbose=0
    )[0][0]

    confidence = float(
        prediction * 100
    )

    if prediction >= 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

        confidence = 100 - confidence

    return (
        sentiment,
        confidence
    )


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    review = "This movie was fantastic"

    sentiment, confidence = predict_sentiment(
        review
    )

    print(
        f"Sentiment: {sentiment}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )