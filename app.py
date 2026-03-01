import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# --------------------------
# Load model and scaler
# --------------------------

model = keras.models.load_model("breast_cancer_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# --------------------------
# Page settings
# --------------------------

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Breast Cancer Prediction App")

st.warning("This app is for educational purposes only and not for medical diagnosis.")

# --------------------------
# Sidebar option
# --------------------------

option = st.sidebar.selectbox(

    "Choose Input Method",

    ("Manual Input (30 Features)", "Upload CSV File")

)

# --------------------------
# Feature names
# --------------------------

feature_names = [
"mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error",
"compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
"worst radius","worst texture","worst perimeter","worst area","worst smoothness",
"worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# ============================================================
# OPTION 1: MANUAL INPUT
# ============================================================

if option == "Manual Input (30 Features)":

    st.subheader("Enter Tumor Feature Values")

    features = []

    col1, col2 = st.columns(2)

    for i, name in enumerate(feature_names):

        if i % 2 == 0:

            value = col1.number_input(name, format="%.5f")

        else:

            value = col2.number_input(name, format="%.5f")

        features.append(value)

    if st.button("Predict"):

        input_data = np.array([features])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        result = np.argmax(prediction)

        prob = np.max(prediction)*100

        st.subheader("Prediction Result")

        if result == 0:

            st.error(f"Malignant (Cancer Detected)\nConfidence: {prob:.2f}%")

        else:

            st.success(f"Benign (No Cancer)\nConfidence: {prob:.2f}%")

# ============================================================
# OPTION 2: CSV UPLOAD
# ============================================================

elif option == "Upload CSV File":

    st.subheader("Upload CSV File")

    st.info("CSV file must contain all 30 feature columns.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")

        st.dataframe(data)

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)

        result = np.argmax(prediction, axis=1)

        probability = np.max(prediction, axis=1)*100

        data["Prediction"] = result

        data["Confidence %"] = probability

        st.subheader("Prediction Results")

        st.dataframe(data)

# --------------------------
# Sidebar info
# --------------------------

st.sidebar.title("About")

st.sidebar.info("""

Breast Cancer Prediction App

Built using:

TensorFlow  
Streamlit  
Neural Network  

Input:

30 Tumor Features

Output:

0 = Malignant  
1 = Benign  

""")
