import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# -----------------------
# Load Model and Scaler
# -----------------------

model = keras.models.load_model("breast_cancer_model.h5")

scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------
# UI Title
# -----------------------

st.title("Breast Cancer Prediction App")

st.write("Enter tumor measurement values below:")

st.header("Tumor Features")

# -----------------------
# Feature Names
# -----------------------

feature_names = [
"mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error",
"compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
"worst radius","worst texture","worst perimeter","worst area","worst smoothness",
"worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# -----------------------
# Input Fields
# -----------------------

features = []

for name in feature_names:
    value = st.number_input(name, format="%.5f")
    features.append(value)

# -----------------------
# Prediction Button
# -----------------------

if st.button("Predict"):

    input_data = np.array([features])

    # Scale input
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    result = np.argmax(prediction)

    probability = np.max(prediction) * 100

    st.subheader("Result:")

    if result == 0:
        st.error(f"Cancer Detected (Malignant)\nConfidence: {probability:.2f}%")
    else:
        st.success(f"No Cancer (Benign)\nConfidence: {probability:.2f}%")