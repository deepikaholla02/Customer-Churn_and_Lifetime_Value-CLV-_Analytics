import streamlit as st
import pandas as pd
import joblib

# Load models and columns

churn_model = joblib.load("models\churn_model.pkl")
clv_model = joblib.load("models\clv_model.pkl")
model_columns = joblib.load("models\model_columns.pkl")

st.set_page_config(page_title="Customer Churn & CLV Analytics", layout="wide")
st.title("📊 Customer Churn & Lifetime Value Analytics")

# ----- User input ------

st.sidebar.header("Customer Details")

def get_input():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.slider("Tenure", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        ),
        "InternetService": st.sidebar.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        ),
        "OnlineSecurity": st.sidebar.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        ),
        "OnlineBackup": st.sidebar.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        ),
        "DeviceProtection": st.sidebar.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        ),
        "TechSupport": st.sidebar.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        ),
        "StreamingTV": st.sidebar.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        ),
        "StreamingMovies": st.sidebar.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        ),
        "Contract": st.sidebar.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        ),
        "PaperlessBilling": st.sidebar.selectbox(
            "Paperless Billing", ["Yes", "No"]
        ),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        ),
        "MonthlyCharges": st.sidebar.number_input("Monthly Charges", min_value=0.0),
        "TotalCharges": st.sidebar.number_input("Total Charges so far",min_value=0.0),
    }
    return pd.DataFrame([data])

input_df = get_input()


# Encode input to match training data
input_encoded = pd.get_dummies(input_df)

# Add missing columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Keep column order
input_encoded = input_encoded[model_columns]


# --- Predict ---

if st.button("Predict"):
    churn_prob = churn_model.predict_proba(input_encoded)[0][1]
    future_clv = clv_model.predict(input_encoded)[0]

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{churn_prob:.2%}")
    col2.metric("Future CLV", f"₹ {future_clv:,.0f}")

    if churn_prob > 0.6:
        st.error("⚠️ High churn risk")
    else:
        st.success("✅ Low churn risk")