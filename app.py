import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Social Media Addiction Checker", layout="wide")

# =========================
# TITLE
# =========================
st.title("📱 Are You Addicted to Social Media?")
st.write("Let's find out using a Decision Tree! 🌳")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Here's what your data looks like:")
    st.dataframe(df.head())

    # =========================
    # USER CONTROLS
    # =========================
    st.sidebar.header("⚙️ Settings")

    test_size = st.sidebar.selectbox(
        "Train-Test Split",
        [0.3, 0.25, 0.2],
        format_func=lambda x: f"{int((1-x)*100)}/{int(x*100)}"
    )

    criterion = st.sidebar.radio("Split Method", ["gini", "entropy"])
    max_depth = st.sidebar.selectbox("Tree Depth", [3, 5, 7, None])

    # =========================
    # TRAIN BUTTON
    # =========================
    if st.button("🚀 Train the Model"):

        # Store original columns before encoding
        original_df = df.copy()
        
        # Encode categorical columns
        label_encoders = {}
        for col in df.select_dtypes(include="object"):
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

        X = df.drop("Addicted_Score", axis=1)
        y = df["Addicted_Score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store in session state
        st.session_state.model = model
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.original_df = original_df

        # =========================
        # RESULTS
        # =========================
        st.success(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")

        st.subheader("📊 Detailed Results")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(cm, cmap='Blues')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.subheader("📈 What Matters Most?")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

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
