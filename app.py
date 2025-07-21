import streamlit as st
import joblib
import numpy as np

# Title and divider
st.title("Salary Prediction App")
st.divider()

# App description
st.write("With this app, you can get estimations for salaries based on experience and job rating.")

# Input fields
years = st.number_input("Enter the years at company: ", value=1, step=1, min_value=0)
jobrate = st.number_input("Enter the Job Rate", value=3.5, step=0.5, min_value=0.0)

# Load trained model
model = joblib.load("linearmodel.pkl")

st.divider()

predict = st.button("Press the button for salary prediction")

st.divider()

if predict:
    st.balloons()
    X1 = np.array([[years, jobrate]])
    prediction = model.predict(X1)
    st.success(f"Salary prediction is ₹{prediction[0]:,.2f}")
else:
    st.info("Please press the button for the app to make predictions")
