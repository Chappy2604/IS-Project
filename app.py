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
    # PREDICTION SECTION
    # =========================
    if 'model' in st.session_state:
        st.markdown("---")
        st.header("🔮 Try It Out - Predict a New Student")
        
        # Get original dataframe
        original_df = st.session_state.original_df
        feature_names = st.session_state.feature_names
        
        # Columns to exclude from input
        excluded_columns = [
            "Student_ID",
        ]
        
        # Filter feature names
        input_feature_names = [col for col in feature_names if col not in excluded_columns]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Enter student info below:")
            
            # Create input fields dynamically
            input_data = {}
            cols = st.columns(2)
            
            for idx, col_name in enumerate(input_feature_names):
                with cols[idx % 2]:
                    # Check if column is categorical
                    if col_name in st.session_state.label_encoders:
                        # Get unique values from original data
                        unique_values = original_df[col_name].unique()
                        input_data[col_name] = st.selectbox(
                            col_name.replace("_", " ").title(),
                            options=unique_values,
                            key=f"input_{col_name}"
                        )
                    else:
                        # Numeric input
                        min_val = float(original_df[col_name].min())
                        max_val = float(original_df[col_name].max())
                        mean_val = float(original_df[col_name].mean())
                        input_data[col_name] = st.number_input(
                            col_name.replace("_", " ").title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"input_{col_name}"
                        )
            
            predict_col1, predict_col2 = st.columns([1, 1])
            
            with predict_col1:
                if st.button("🎯 Predict Addiction Level", use_container_width=True):
                    # Fill in excluded columns with default values
                    full_input = {}
                    for col_name in feature_names:
                        if col_name in input_data:
                            full_input[col_name] = input_data[col_name]
                        else:
                            # Use the most common value or mean for excluded columns
                            if col_name in st.session_state.label_encoders:
                                full_input[col_name] = original_df[col_name].mode()[0]
                            else:
                                full_input[col_name] = original_df[col_name].mean()
                    
                    # Encode categorical inputs
                    encoded_input = {}
                    for col_name, value in full_input.items():
                        if col_name in st.session_state.label_encoders:
                            if col_name not in input_data:
                                # For excluded columns, use encoded value directly
                                encoded_input[col_name] = st.session_state.label_encoders[col_name].transform([value])[0]
                            else:
                                encoded_input[col_name] = st.session_state.label_encoders[col_name].transform([value])[0]
                        else:
                            encoded_input[col_name] = value
                    
                    # Create dataframe for prediction
                    input_df = pd.DataFrame([encoded_input])
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_df)[0]
                    
                    # Display result
                    st.session_state.prediction_result = prediction
            
            with predict_col2:
                if st.button("🔄 Reset Form", use_container_width=True):
                    # Clear all input fields by rerunning
                    for key in list(st.session_state.keys()):
                        if key.startswith('input_'):
                            del st.session_state[key]
                    if 'prediction_result' in st.session_state:
                        del st.session_state['prediction_result']
                    st.rerun()
        
        with col2:
            if 'prediction_result' in st.session_state:
                prediction = st.session_state.prediction_result
                
                st.markdown("### 📊 Prediction Result")
                
                # Display score with rating
                st.metric("Addiction Score", f"{prediction}/5")
                
                # Rating and message based on score
                if prediction == 0:
                    st.success("### ⭐⭐⭐⭐⭐ EXCELLENT!")
                    st.write("**Status:** Not Addicted at all")
                    st.write("🎉 Great job! You have a very healthy relationship with social media!")
                    st.balloons()
                elif prediction == 1:
                    st.success("### ⭐⭐⭐⭐ GOOD!")
                    st.write("**Status:** Very Low Addiction")
                    st.write("👍 You're doing well! Keep maintaining this healthy balance!")
                elif prediction == 2:
                    st.info("### ⭐⭐⭐ OKAY")
                    st.write("**Status:** Mild Addiction")
                    st.write("😊 Not bad, but there's room for improvement. Try reducing screen time a bit!")
                elif prediction == 3:
                    st.warning("### ⭐⭐ WARNING")
                    st.write("**Status:** Moderate Addiction")
                    st.write("⚠️ You might want to watch your social media usage. Consider setting limits!")
                elif prediction == 4:
                    st.error("### ⭐ CONCERNING")
                    st.write("**Status:** High Addiction")
                    st.write("🚨 This is concerning. Try to reduce your social media time significantly!")
                else:  # prediction == 5
                    st.error("### ❌ CRITICAL")
                    st.write("**Status:** Severe Addiction")
                    st.write("🆘 This is serious! Consider seeking help or taking a digital detox!")
                
                # Progress bar visual
                st.progress(prediction / 5)
