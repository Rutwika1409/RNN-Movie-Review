"""
train.py

This script:
1. Loads the IMDB dataset.
2. Preprocesses the data.
3. Builds a SimpleRNN model.
4. Trains the model.
5. Evaluates performance.
6. Saves the trained model.
"""

import os

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense
)


# =====================================================
# Create Models Directory
# =====================================================

os.makedirs("../models", exist_ok=True)


# =====================================================
# Parameters
# =====================================================

VOCAB_SIZE = 10000

MAX_LENGTH = 200

EMBEDDING_DIM = 32


# =====================================================
# Load Dataset
# =====================================================

print("Loading IMDB Dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# =====================================================
# Pad Sequences
# =====================================================

print("\nPadding Sequences...")

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# =====================================================
# Build SimpleRNN Model
# =====================================================

print("\nBuilding Model...")

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    )
)

model.add(
    SimpleRNN(
        units=64,
        activation="tanh"
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)

model.summary()


# =====================================================
# Compile Model
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# Train Model
# =====================================================

print("\nTraining Model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=5,
    batch_size=64
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
# Save Model
# =====================================================

model_path = "../models/imdb_rnn.keras"

model.save(
    model_path
)

print(
    f"\nModel Saved Successfully:\n{model_path}"
)