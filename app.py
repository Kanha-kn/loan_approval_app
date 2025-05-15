import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load model and scaler
model = load_model("loan_model.h5")  # Save your trained model with this name
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Loan Approval Prediction Using ANN")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to DataFrame
data = {
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Gender_Male': 1 if Gender == "Male" else 0,
    'Married_Yes': 1 if Married == "Yes" else 0,
    'Education_Not Graduate': 1 if Education == "Not Graduate" else 0,
    'Self_Employed_Yes': 1 if Self_Employed == "Yes" else 0,
    'Property_Area_Semiurban': 1 if Property_Area == "Semiurban" else 0,
    'Property_Area_Urban': 1 if Property_Area == "Urban" else 0
}
input_df = pd.DataFrame([data])

# Scale numeric data
scaled_input = scaler.transform(input_df)

if st.button("Predict"):
    result = model.predict(scaled_input)
    output = "Loan Approved" if result[0][0] > 0.5 else "Loan Rejected"
    st.success(f"Prediction: {output}")
