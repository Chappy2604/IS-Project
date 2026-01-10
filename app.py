import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    with open("addiction_model (2).pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# ------------------------------
# App UI
# ------------------------------
st.set_page_config(
    page_title="Student Addiction Score Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Student Addiction Score Predictor")
st.markdown("Predict student addiction score, probability, and risk level.")

st.divider()

# ------------------------------
# Input Fields (MATCH TRAINING COLUMNS)
# ------------------------------
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=10, max_value=100, value=21)
Sleep_Hours = st.slider("Sleep Hours per Day", 0, 12, 6)
Study_Hours = st.slider("Study Hours per Day", 0, 12, 3)
Social_Media_Usage = st.selectbox(
    "Social Media Usage",
    ["Low", "Medium", "High"]
)

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔍 Predict Addiction Score"):

    input_data = {
        "Gender": Gender,
        "Age": Age,
        "Sleep_Hours": Sleep_Hours,
        "Study_Hours": Study_Hours,
        "Social_Media_Usage": Social_Media_Usage
    }

    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability (works if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = np.max(probabilities)
    else:
        confidence = None

    # ------------------------------
    # Risk Level Logic (EDIT if needed)
    # ------------------------------
    if prediction <= 3:
        risk = "Low"
        color = "green"
    elif prediction <= 6:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    # ------------------------------
    # Display Results
    # ------------------------------
    st.success(f"🎯 Predicted Addiction Score: **{prediction}**")

    if confidence is not None:
        st.write(f"📈 Prediction Confidence: **{confidence * 100:.2f}%**")
        st.progress(int(confidence * 100))

    st.markdown(
        f"### 🚦 Risk Level: <span style='color:{color}'>{risk}</span>",
        unsafe_allow_html=True
    )

# ------------------------------
# Footer
# ------------------------------
st.caption("Machine Learning Model • OneHotEncoder • Streamlit")
