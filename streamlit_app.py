# streamlit_app.py
import streamlit as st
import requests
import numpy as np

st.title('Churn Prediction App')

# Input fields for features
st.write("Enter feature values:")
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add more features as needed

if st.button('Predict'):
    features = [feature1, feature2]  # Add more features as needed
    try:
        response = requests.post(
            'http://127.0.0.1:5000/predict', 
            json={'features': features}
        )
        prediction = response.json()
        st.write(f'Prediction: {prediction["prediction"]}')
    except Exception as e:
        st.write(f'Error: {e}')
