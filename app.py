import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# Load model and scaler
model = keras.models.load_model("breast_cancer_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Breast Cancer Prediction App")

st.warning("This app is for educational purposes only.")

st.header("Tumor Features")

# Feature names
feature_names = [
"mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error",
"compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
"worst radius","worst texture","worst perimeter","worst area","worst smoothness",
"worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# Sample benign data
sample_benign = [
12.34,14.54,78.94,468.5,0.098,0.072,0.017,0.012,0.190,0.059,
0.25,1.20,1.65,20.5,0.006,0.018,0.021,0.009,0.015,0.002,
13.45,16.67,87.12,550.3,0.134,0.145,0.095,0.065,0.250,0.075
]

# Sample malignant data
sample_malignant = [
17.99,10.38,122.8,1001,0.118,0.277,0.300,0.147,0.242,0.078,
1.095,0.905,8.589,153.4,0.006,0.049,0.053,0.015,0.030,0.006,
25.38,17.33,184.6,2019,0.162,0.665,0.711,0.265,0.460,0.118
]

# Buttons
col1, col2 = st.columns(2)

if col1.button("Use Benign Sample"):
    input_data = np.array([sample_benign])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    result = np.argmax(prediction)
    prob = np.max(prediction)*100

    st.subheader("Result")
    st.success(f"No Cancer (Benign) — Confidence: {prob:.2f}%")

if col2.button("Use Malignant Sample"):
    input_data = np.array([sample_malignant])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    result = np.argmax(prediction)
    prob = np.max(prediction)*100

    st.subheader("Result")
    st.error(f"Cancer Detected (Malignant) — Confidence: {prob:.2f}%")

st.divider()

# Manual input
st.subheader("Or Enter Values Manually")

features = []

for name in feature_names:
    val = st.number_input(name, format="%.5f")
    features.append(val)

if st.button("Predict Manually"):

    input_data = np.array([features])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    result = np.argmax(prediction)
    prob = np.max(prediction)*100

    st.subheader("Result")

    if result == 0:
        st.error(f"Cancer Detected (Malignant) — Confidence: {prob:.2f}%")
    else:
        st.success(f"No Cancer (Benign) — Confidence: {prob:.2f}%")
