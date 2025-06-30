import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. Load the trained model pipeline ---
# Ensure the path to your saved model is correct relative to this script.
# The model was saved in 'models/titanic_survival_predictor_pipeline.joblib'
model_path = 'models/titanic_survival_predictor_pipeline.joblib'

try:
    model_pipeline = joblib.load(model_path)
    st.success("Machine Learning Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure it is saved correctly.")
    st.info("You might need to run the 'explore.ipynb' notebook to train and save the model first.")
    st.stop() # Stop the app if the model can't be loaded
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.title("Titanic Survival Predictor")
st.markdown("---")
st.write("Enter passenger details to predict their survival on the Titanic.")

# --- 2. Create input widgets for passenger features ---
# These inputs correspond to the features your model was trained on:
# 'Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Sex', 'Embarked'

st.header("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30) # Min, Max, Default
    sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)

with col2:
    parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Fare ($)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])


# --- 3. Prediction Button ---
if st.button("Predict Survival"):
    # Create a DataFrame from user inputs, matching the original training features
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])

    st.write("---")
    st.subheader("Prediction Result:")

    try:
        # Make prediction using the loaded pipeline
        # The pipeline handles all preprocessing (scaling, one-hot encoding) internally
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)[0]

        if prediction == 1:
            st.success(f"**Survived!** 🎉 (Confidence: {prediction_proba[1]*100:.2f}%)")
            st.balloons() # Add a fun animation for survival
        else:
            st.error(f"**Did Not Survive.** 😔 (Confidence: {prediction_proba[0]*100:.2f}%)")

        st.write("---")
        st.subheader("Input Data Summary:")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check your input values.")

st.markdown("---")
st.caption("Developed as part of the 4Geeks Academy ML Web App Tutorial.")