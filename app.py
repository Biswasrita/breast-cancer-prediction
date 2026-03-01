import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = keras.models.load_model("breast_cancer_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Breast Cancer Prediction App")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Model Info")

st.sidebar.info("""
Model: Neural Network  
Dataset: Breast Cancer Wisconsin  
Features: 30 Tumor Measurements  

Prediction Output:

0 = Malignant (Cancer)  
1 = Benign (No Cancer)  

""")

# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
"mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error",
"compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
"worst radius","worst texture","worst perimeter","worst area","worst smoothness",
"worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# ============================================================
# FEATURE DESCRIPTION
# ============================================================

with st.expander("About the 30 Features"):

    st.write("""

These features are calculated from breast tumor biopsy images.

They describe tumor characteristics:

• Radius → tumor size  

• Texture → pixel variation  

• Perimeter → tumor boundary  

• Area → tumor area  

• Smoothness → surface smoothness  

• Compactness → shape complexity  

• Concavity → tumor concave regions  

• Symmetry → tumor symmetry  

Worst = worst value  

Mean = average  

Error = variation  

These features help predict cancer.

""")

# ============================================================
# INPUT METHOD
# ============================================================

option = st.radio(

"Choose Input Method",

("Manual Input", "Upload CSV File")

)

# ============================================================
# MANUAL INPUT
# ============================================================

if option == "Manual Input":

    st.subheader("Enter Tumor Features")

    features = []

    col1, col2 = st.columns(2)

    for i, name in enumerate(feature_names):

        if i % 2 == 0:

            val = col1.number_input(name, format="%.5f")

        else:

            val = col2.number_input(name, format="%.5f")

        features.append(val)

    if st.button("Predict"):

        input_data = np.array([features])

        scaled = scaler.transform(input_data)

        prediction = model.predict(scaled)

        result = np.argmax(prediction)

        confidence = np.max(prediction)*100

        st.subheader("Prediction Result")

        if result == 0:

            st.error("Malignant (Cancer Detected)")

        else:

            st.success("Benign (No Cancer)")

        st.write(f"Confidence: {confidence:.2f}%")

        # GRAPH

        chart = pd.DataFrame({

            "Confidence":[confidence]

        })

        st.bar_chart(chart)

# ============================================================
# CSV INPUT
# ============================================================

if option == "Upload CSV File":

    st.subheader("Upload CSV")

    file = st.file_uploader("Choose CSV file")

    if file is not None:

        data = pd.read_csv(file)

        st.write("Uploaded Data")

        st.dataframe(data)

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)

        result = np.argmax(prediction, axis=1)

        confidence = np.max(prediction, axis=1)*100

        data["Prediction"] = result

        data["Confidence %"] = confidence

        st.subheader("Prediction Result")

        st.dataframe(data)

        # PREDICTION GRAPH

        st.subheader("Prediction Graph")

        count = data["Prediction"].value_counts()

        st.bar_chart(count)

        # CONFIDENCE GRAPH

        st.subheader("Confidence Graph")

        st.line_chart(data["Confidence %"])

        # DOWNLOAD

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(

            "Download Results",

            csv,

            "prediction_result.csv",

            "text/csv"

        )
