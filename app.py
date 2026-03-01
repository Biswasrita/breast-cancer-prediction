import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow import keras

# -----------------------------
# Load model and scaler
# -----------------------------

model = keras.models.load_model("breast_cancer_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Page config
# -----------------------------

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("Breast Cancer Prediction App")

st.warning("For educational purposes only. Not for medical use.")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Options")

option = st.sidebar.selectbox(

    "Choose Prediction Method",

    ("Use Sample Data", "Upload CSV File")

)

# -----------------------------
# Feature Names
# -----------------------------

feature_names = [
"mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error",
"compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
"worst radius","worst texture","worst perimeter","worst area","worst smoothness",
"worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# -----------------------------
# SAMPLE DATA OPTION
# -----------------------------

if option == "Use Sample Data":

    st.subheader("Click button to test prediction")

    sample = [
    17.99,10.38,122.8,1001,0.118,0.277,0.300,0.147,0.242,0.078,
    1.095,0.905,8.589,153.4,0.006,0.049,0.053,0.015,0.030,0.006,
    25.38,17.33,184.6,2019,0.162,0.665,0.711,0.265,0.460,0.118
    ]

    if st.button("Predict Sample"):

        input_data = np.array([sample])

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        result = np.argmax(prediction)

        prob = np.max(prediction)*100

        st.subheader("Result")

        if result == 0:

            st.error(f"Cancer Detected (Malignant)\nConfidence: {prob:.2f}%")

        else:

            st.success(f"No Cancer (Benign)\nConfidence: {prob:.2f}%")

# -----------------------------
# CSV UPLOAD OPTION
# -----------------------------

elif option == "Upload CSV File":

    st.subheader("Upload CSV file with 30 features")

    file = st.file_uploader("Choose CSV file")

    if file is not None:

        data = pd.read_csv(file)

        st.write("Uploaded Data")

        st.dataframe(data)

        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)

        result = np.argmax(prediction, axis=1)

        data["Prediction"] = result

        st.subheader("Prediction Result")

        st.dataframe(data)

# -----------------------------
# Info section
# -----------------------------

st.sidebar.info("""

This app predicts breast cancer using Neural Network.

Input: Tumor features

Output:

0 = Malignant

1 = Benign

""")
