
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
features = joblib.load("features.pkl")

# Autofill presets
autofill_presets = {
    "Regular": [1, 1, 3, 100.0, 150.0, 20.0, 15.0, 5, 2.0, 1],
    "Terminal": [2, 2, 4, 110.0, 180.0, 25.0, 18.0, 6, 2.0, 2],
    "Interchange": [3, 3, 5, 120.0, 200.0, 30.0, 20.0, 7, 3.0, 3],
    "Custom": [None] * len(features)
}

# UI
st.title("🚇 Metro Station Civil Cost Estimator")
st.write("Estimate metro station civil construction cost using 10 key design parameters.")

station_type = st.selectbox("Select Station Type:", list(autofill_presets.keys()))
preset_values = autofill_presets[station_type]

st.subheader("🧾 Input Parameters")
input_data = {}
for i, feature in enumerate(features):
    default = preset_values[i]
    input_data[feature] = st.number_input(feature, value=default if default is not None else 0.0)

# Predict cost
if st.button("💰 Predict Civil Cost"):
    try:
        df_input = pd.DataFrame([input_data])
        df_preprocessed = preprocessor.transform(df_input)
        prediction = model.predict(df_preprocessed)[0]
        st.success(f"Estimated Civil Cost: ₹{prediction:,.2f} crore")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Batch prediction
st.subheader("📂 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])

if uploaded_file:
    try:
        batch_df = pd.read_excel(uploaded_file)
        batch_preprocessed = preprocessor.transform(batch_df)
        batch_predictions = model.predict(batch_preprocessed)
        batch_df["Predicted Civil Cost (Cr)"] = batch_predictions
        st.write("✅ Prediction Results")
        st.dataframe(batch_df)
        st.download_button("📥 Download Predictions", data=batch_df.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
