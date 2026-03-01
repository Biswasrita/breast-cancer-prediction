#  Breast Cancer Prediction App

A Machine Learning web application that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using a Neural Network model.

---

##  Live Demo

 **Click here to use the app:**  
https://breast-cancer-prediction-ms25nioupnxxas4ssyxeuk.streamlit.app/

---

##  Project Overview

This application uses a trained **Artificial Neural Network (ANN)** model to analyze **30 tumor features** obtained from breast biopsy data and predict cancer.

The app is built with:

- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

##  Features

✔ Manual Input (30 Features)  
✔ CSV File Upload Prediction  
✔ Batch Prediction Support  
✔ Confidence Score Display  
✔ Graph Visualization  
✔ Download Prediction Results  
✔ Interactive UI  

---

##  Input Features

The model uses **30 medical features**, including:

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

These features describe tumor characteristics extracted from biopsy images.

---

##  Output

The model predicts:

- **Malignant (Cancer)**
- **Benign (No Cancer)**

With confidence score.

---

##  Model Information

Model Type: Neural Network  
Dataset: Breast Cancer Wisconsin Dataset  
Framework: TensorFlow / Keras  

---

##  How to Run Locally

### Step 1: Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-prediction.git
