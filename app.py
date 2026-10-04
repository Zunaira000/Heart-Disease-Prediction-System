"""
Heart Disease Prediction System - Streamlit App
------------------------------------------------
Takes 4 inputs, loads the trained model (heart_model.pkl) and predicts
whether heart disease is likely present (1) or not (0).

No label encoding is used: all 4 inputs are already numbers.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter the 4 patient values below and the trained **Logistic Regression** model
    will predict whether heart disease is likely.

    *Trained on the Cleveland Clinic heart disease dataset (297 patients).
    Accuracy on held-out test data: **83%**.*
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    ca = st.selectbox(
        "Major vessels coloured by fluoroscopy (ca)",
        [0, 1, 2, 3],
        help="Number of major blood vessels (0-3) seen in the fluoroscopy test.",
    )
    thalach = st.number_input(
        "Maximum heart rate achieved (thalach)",
        min_value=60, max_value=220, value=150, step=1,
        help="Highest heart rate reached during the exercise test (beats per minute).",
    )
with col2:
    oldpeak = st.number_input(
        "ST depression (oldpeak)",
        min_value=0.0, max_value=7.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest.",
    )
    exang_label = st.selectbox(
        "Exercise-induced angina (exang)",
        ["No", "Yes"],
        help="Did exercise cause chest pain?",
    )

exang = 1 if exang_label == "Yes" else 0  # already a 0/1 number, not label encoding

if st.button("Predict", type="primary", use_container_width=True):
    # Column names and order must match training: ca, oldpeak, thalach, exang
    features = pd.DataFrame(
        [[ca, oldpeak, thalach, exang]],
        columns=["ca", "oldpeak", "thalach", "exang"],
    )

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Heart disease LIKELY")
    else:
        st.success("### ✅ Heart disease UNLIKELY")
    st.metric("Estimated probability of heart disease", f"{prob:.0%}")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(features, hide_index=True)

st.divider()
st.caption(
    "⚠️ Educational project only - NOT a medical diagnosis. "
    "Always consult a qualified doctor. Data: UCI Heart Disease (Cleveland) - "
    "Detrano, Janosi, Steinbrunn, Pfisterer et al."
)
