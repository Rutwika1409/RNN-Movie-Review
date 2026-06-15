"""
app.py

Streamlit frontend for IMDB Sentiment Analysis.

Users can:
1. Enter a movie review.
2. Predict sentiment.
3. View confidence score.
"""

import streamlit as st

from src.predict import predict_sentiment


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    """
    Dataset:
    IMDB Movie Reviews

    Model:
    SimpleRNN

    Framework:
    TensorFlow / Keras

    Frontend:
    Streamlit
    """
)


# =====================================================
# Title
# =====================================================

st.title(
    "🎬 IMDB Sentiment Analysis"
)

st.write(
    """
    Enter a movie review below and the RNN model
    will predict whether the review is positive
    or negative.
    """
)


# =====================================================
# Review Input
# =====================================================

review = st.text_area(
    "Enter Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)


# =====================================================
# Prediction Button
# =====================================================

if st.button(
    "Analyze Sentiment"
):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        with st.spinner(
            "Analyzing Review..."
        ):

            sentiment, confidence = (
                predict_sentiment(
                    review
                )
            )

        st.success(
            "Analysis Complete"
        )

        st.subheader(
            "Prediction"
        )

        if sentiment == "Positive":

            st.success(
                f"Sentiment: {sentiment}"
            )

        else:

            st.error(
                f"Sentiment: {sentiment}"
            )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )


# =====================================================
# Footer
# =====================================================

st.write("---")

st.caption(
    "Built using TensorFlow, SimpleRNN and Streamlit"
)