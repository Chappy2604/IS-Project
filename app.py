import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    with open("addiction_model.pkl", "rb") as file:
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
Age = st.number_input("Age", min_value=10, max_value=100, value=21)
Gender = st.selectbox("Gender", ["Male", "Female"])
Academic_Level = st.selectbox(
    "Academic Level", ["Undergraduate", "Graduate", "High School"]
)
Country = st.selectbox("Country", ["Afghanistan", "Albania", "Andorra", "Argentina", "Armenia", "Australia", "Austria", 
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Belgium", "Bhutan", 
    "Bolivia", "Bosnia", "Brazil", "Bulgaria", "Canada", "Chile", "China", "Colombia", 
    "Costa Rica", "Croatia", "Cyprus", "Czech Rep", "Denmark", "Ecuador", "Egypt", 
    "Estonia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece", 
    "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iraq", "Ireland", 
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kosovo", 
    "Kuwait", "Kyrgyzstan", "Latvia", "Lebanon", "Liechtenstein", "Lithuania", 
    "Luxembourg", "Malaysia", "Maldives", "Malta", "Mexico", "Moldova", "Monaco", 
    "Montenegro", "Morocco", "Nepal", "Netherlands", "New Zealand", "Nigeria", 
    "North Macedonia", "Norway", "Oman", "Pakistan", "Panama", "Paraguay", "Peru", 
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "San Marino", 
    "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa", "South Korea", 
    "Spain", "Sri Lanka", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", 
    "Thailand", "Trinidad", "Turkey", "UAE", "UK", "Ukraine", "Uruguay", "USA", 
    "Uzbekistan", "Vatican City", "Venezuela", "Vietnam", "Yemen"])
Avg_Daily_Usage_Hours = st.slider("Average Daily Usage Hours", 0, 24, 5)
Most_Used_Platform = st.selectbox(
    "Most Used Platform", ["LinkedIn", "TikTok", "Instagram", "Facebook", "Snapchat", "WhatsApp", "WeChat", "Twitter", "YouTube", "LINE", "VKontakte", "KakaoTalk"]
)
Affects_Academic_Performance = st.selectbox(
    "Affects Academic Performance", ["Yes", "No"]
)
Sleep_Hours_Per_Night = st.slider("Sleep Hours Per Night", 0, 12, 6)
Mental_Health_Score = st.slider("Mental Health Score", 0, 10, 5)
Relationship_Status = st.selectbox("Relationship Status", ["Single", "In a Relationship", "Complicated"])
Conflicts_Over_Social_Media = st.selectbox(
    "Conflicts Over Social Media", ["Yes", "No"]
)

input_data = {
    "Age": Age,
    "Gender": Gender,
    "Academic_Level": Academic_Level,
    "Country": Country,
    "Avg_Daily_Usage_Hours": Avg_Daily_Usage_Hours,
    "Most_Used_Platform": Most_Used_Platform,
    "Affects_Academic_Performance": Affects_Academic_Performance,
    "Sleep_Hours_Per_Night": Sleep_Hours_Per_Night,
    "Mental_Health_Score": Mental_Health_Score,
    "Relationship_Status": Relationship_Status,
    "Conflicts_Over_Social_Media": Conflicts_Over_Social_Media
}

input_df = pd.DataFrame([input_data])

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔍 Predict Addiction Score"):

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
