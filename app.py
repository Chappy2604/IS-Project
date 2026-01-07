import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Social Media Addiction Checker", layout="wide")

# =========================
# TITLE
# =========================
st.title("📱 Are You Addicted to Social Media?")
st.write("Let's find out using a Decision Tree! 🌳")

# =========================
# LOAD THE PRE-TRAINED MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        with open("addiction_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data = load_model()

if data is None:
    st.error("⚠️ 'addiction_model.pkl' not found. Please upload it to the Colab folder.")
    st.stop()

model = data["model"]
encoders = data["label_encoders"]
features = data["feature_names"]

st.title("📱 Social Media Addiction Predictor")
st.info("Model loaded successfully from local storage.")

# =========================
# DYNAMIC INPUT FORM
# =========================
st.subheader("Enter User Details")
input_dict = {}

# Create 2 columns for a cleaner UI
col1, col2 = st.columns(2)

for i, col_name in enumerate(features):
    # Skip ID columns if they exist
    if "ID" in col_name.upper(): continue
        
    with col1 if i % 2 == 0 else col2:
        if col_name in encoders:
            # Use the encoder's classes for the selectbox
            options = encoders[col_name].classes_
            val = st.selectbox(col_name.replace("_", " "), options)
            input_dict[col_name] = encoders[col_name].transform([val])[0]
        else:
            # Standard number input for numerical data
            val = st.number_input(col_name.replace("_", " "), value=0.0)
            input_dict[col_name] = val

# =========================
# PREDICTION
# =========================
if st.button("🎯 Run Prediction", use_container_width=True):
    # Convert input dict to 2D array for the model
    input_df = pd.DataFrame([input_dict])[features] 
    prediction = model.predict(input_df)[0]
    
    st.divider()
    st.metric("Predicted Addiction Score", f"{prediction}/5")
    
    # Simple logic for feedback
    if prediction >= 4:
        st.error("Status: High Risk of Addiction")
    elif prediction >= 2:
        st.warning("Status: Moderate Usage")
    else:
        st.success("Status: Healthy Usage")
