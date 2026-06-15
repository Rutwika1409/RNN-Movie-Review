"""
evaluate.py

This script:
1. Loads the trained RNN model.
2. Loads the IMDB dataset.
3. Evaluates model performance.
4. Generates classification metrics.
5. Displays a confusion matrix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# Configuration
# =====================================================

VOCAB_SIZE = 10000

MAX_LENGTH = 200


# =====================================================
# Load Dataset
# =====================================================

print("Loading IMDB Dataset...")

(_, _), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)


# =====================================================
# Pad Sequences
# =====================================================

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH
)


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

print("\nLoading Model...")
print(MODEL_PATH)

model = load_model(
    MODEL_PATH
)


# =====================================================
# Evaluate Model
# =====================================================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# =====================================================
# Generate Predictions
# =====================================================

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = (
    predictions >= 0.5
).astype(int).flatten()


# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Negative",
            "Positive"
        ]
    )
)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Positive"
    ]
)

disp.plot()

plt.title(
    "IMDB Sentiment Analysis Confusion Matrix"
)

plt.show()