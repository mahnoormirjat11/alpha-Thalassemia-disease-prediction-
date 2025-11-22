import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("models/alpha_rf.pkl")
meta = joblib.load("models/meta.pkl")

st.title("🩸 Alpha-Thalassemia Prediction App")

st.write("Upload CBC data to get prediction.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    prediction = model.predict(df)
    
    # Decode label if mapping exists
    if meta["label_map"]:
        inv_map = {v:k for k,v in meta["label_map"].items()}
        decoded = [meta["label_map"][p] for p in prediction]
        st.write("### Prediction:", decoded)
    else:
        st.write("### Prediction:", prediction)
