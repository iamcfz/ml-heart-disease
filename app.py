# app.py

import streamlit as st
import joblib
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("/Users/toyinmalomo/DDSC-heartdisease/heart_model.pkl")

model = load_model()

st.set_page_config(page_title="Heart Disease Predictor")

st.title("Heart Disease Risk Predictor")

st.markdown("Enter the patient information below:")

# User Inputs
age = st.number_input("Age", 0, 120)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (1-4)", 1, 4, help="Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic")
trestbps = st.number_input("Resting Blood Pressure", 0, 300, help="Resting blood pressure (in mm Hg on admission to the hospital)")
chol = st.number_input("Cholesterol", 0, 600, help="A healthy cholesterol level is usually below 200 mg/dl. Values above 240 are considered high")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = True; 0 = False")
restecg = st.number_input("Resting ECG (0-2)", 0, 2, help="Value 0: Normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria")
thalach = st.number_input("Max Heart Rate Achieved", 0, 250)
exang = st.selectbox("Exercise Induced Angina", [0, 1], help="Do you experience chest pain during exercise or physical activity?")
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, step=0.1, help="This measures how the heart's electrical activity changes under stress. (Higher numbers usually indicate more strain on the heart)")
slope = st.number_input("Slope (1-3)", 1, 3, help = "Value 1: Upsloping, Value 2: flat, Value 3: Downward sloping")
ca = st.number_input("Number of Major Vessels (0-3)", 0, 3)
thal = st.selectbox("Thalassemia", [3, 6, 7], help = "3 = Normal, 6 = Fixed Defect, 7 = Reversable Defect")

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    confidence = round(probability * 100, 2)

    if prediction == 1:
        st.error(f"HIGH RISK of heart disease (Confidence: {confidence}%)")
    else:
        st.success(f"LOW RISK of heart disease (Confidence: {confidence}%)")