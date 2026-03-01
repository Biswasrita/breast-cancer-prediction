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

st.warning("This app is for educational purposes only and not medical diagnosis.")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Model Info")

st.sidebar.metric("Model Accuracy", "97.8%")

st.sidebar.info("""
Model: Neural Network  
Dataset: Breast Cancer Wisconsin  
Features: 30 Tumor Measurements  
Output:
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
# FEATURE DESCRIPTION SECTION
# ============================================================

with st.expander("What are these 30 features?"):

    st.write("""

These features are calculated from a breast tumor biopsy image.

They describe characteristics of tumor cells:

• Radius → Size of tumor

• Texture → Variation in pixel intensity

• Perimeter → Boundary length

• Area → Tumor area

• Smoothness → How smooth tumor surface is

• Compactness → Shape complexity

• Concavity → Depth of concave portions

• Concave points → Number of concave regions

• Symmetry → Tumor symmetry

• Fractal dimension → Shape irregularity

Mean = average value  
Error = variation  
Worst = worst case value  

These help AI detect cancer.

""")

# ============================================================
# SELECT OPTION
# ============================================================

option = st.radio(

"Choose Input Method",

("Manual Input", "Upload CSV File")

)

# ============================================================
# MANUAL INPUT SECTION
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

        st.subheader("Result")

        if result == 0:

            st.error(f"Malignant (Cancer Detected)")

        else:

            st.success(f"Benign (No Cancer)")

        st.write(f"Confidence: {confidence:.2f}%")

        # GRAPH

        st.subheader("Confidence Graph")

        chart_data = pd.DataFrame({

            "Result":["Confidence"],

            "Value":[confidence]

        })

        st.bar_chart(chart_data.set_index("Result"))

# ============================================================
# CSV UPLOAD SECTION
# ============================================================

if option == "Upload CSV File":

    st.subheader("Upload CSV")

    file = st.file_uploader("Upload file")

    if file is not None:

        data = pd.read_csv(file)

        st.write("Input Data")

        st.dataframe(data)

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)

        result = np.argmax(prediction, axis=1)

        confidence = np.max(prediction, axis=1)*100

        data["Prediction"] = result

        data["Confidence %"] = confidence

        st.subheader("Prediction Result")

        st.dataframe(data)

        # BAR GRAPH

        st.subheader("Prediction Graph")

        count = data["Prediction"].value_counts()

        st.bar_chart(count)

        # CONFIDENCE GRAPH

        st.subheader("Confidence Graph")

        st.line_chart(data["Confidence %"])

        # DOWNLOAD BUTTON

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(

            "Download Result",

            csv,

            "prediction_result.csv",

            "text/csv"

        )
