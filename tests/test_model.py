"""
test_model.py

Automated tests for IMDB RNN Project.

Tests:
1. Model loads correctly.
2. Prediction output shape is valid.
3. Probability is between 0 and 1.
"""

import os
import numpy as np

from tensorflow.keras.models import load_model


# =====================================================
# Load Model Path
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


# =====================================================
# Test Model Loading
# =====================================================

def test_model_load():

    model = load_model(
        MODEL_PATH
    )

    assert model is not None


# =====================================================
# Test Prediction Shape
# =====================================================

def test_prediction_shape():

    model = load_model(
        MODEL_PATH
    )

    sample_input = np.random.randint(
        0,
        10000,
        size=(1, 200)
    )

    prediction = model.predict(
        sample_input,
        verbose=0
    )

    assert prediction.shape == (1, 1)


# =====================================================
# Test Probability Range
# =====================================================

def test_probability_range():

    model = load_model(
        MODEL_PATH
    )

    sample_input = np.random.randint(
        0,
        10000,
        size=(1, 200)
    )

    prediction = model.predict(
        sample_input,
        verbose=0
    )[0][0]

    assert 0 <= prediction <= 1