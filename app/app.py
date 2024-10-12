import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = os.path.join(os.getcwd(), 'best_model_pipeline.joblib')
best_pipeline = joblib.load(model_path)

# Load the training data
data_path = os.path.join(os.getcwd(), 'training_data.csv')
data = pd.read_csv(data_path)

st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox('Brand', data['Company'].unique())
type_name = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 8, 16, 32, 64])
os = st.selectbox('Operating System', data['OpSys'].unique())
weight = st.number_input('Weight of the laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)
ppi = st.number_input('Pixels Per Inch (PPI)', min_value=50, max_value=500, step=1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
cpu_brand = st.selectbox('CPU Brand', data['CPU_brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])
gpu_brand = st.selectbox('GPU Brand', data['GPU_brand'].unique())

if st.button('Predict Price'):
    # Prepare input data
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'OpSys': [os],
        'Weight': [weight],
        'PPI': [ppi],
        'TouchScreen': [1 if touchscreen == 'Yes' else 0],
        'IPS': [1 if ips == 'Yes' else 0],
        'CPU_brand': [cpu_brand],
        'HDD': [hdd],
        'SSD': [ssd],
        'GPU_brand': [gpu_brand]
    })
    
    # Make prediction
    predicted_log_price = best_pipeline.predict(input_data)
    predicted_price = np.exp(predicted_log_price)[0]

    # Calculate error margin
    std_dev_errors = 0.1334  # Given standard deviation of errors
    error_margin = std_dev_errors * 1.96  # 95% confidence interval
    lower_bound = np.exp(predicted_log_price - error_margin)[0]
    upper_bound = np.exp(predicted_log_price + error_margin)[0]

    # Display the predicted price in USD with comma as thousand separator
    st.success("Predicted price for the new laptop: ${:.2f} ± ${:.2f}".format(
        predicted_price, predicted_price * error_margin))
