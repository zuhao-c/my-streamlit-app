import streamlit as st
import joblib
import pandas as pd

# Load the model and preprocessing objects
model = joblib.load('../models/hypertuning_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
pca = joblib.load('../models/pca.pkl')

# Function to preprocess the input
def preprocess_input(data):
    # Scale numerical features
    scaled_data = scaler.transform(data)
    
    # Apply PCA
    pca_data = pca.transform(scaled_data)
    return pca_data

# Streamlit app title
st.title("Customer Churn Prediction App")

# User inputs for the features
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0)
gender = st.radio("Gender", ["Female", "Male"])
senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
partner = st.radio("Partner", ["No", "Yes"])
dependents = st.radio("Dependents", ["No", "Yes"])
phone_service = st.radio("Phone Service", ["No", "Yes"])
multiple_lines = st.radio("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.radio("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.radio("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.radio("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.radio("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.radio("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.radio("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
payment_method = st.radio("Payment Method", [
    "Bank transfer (automatic)", 
    "Credit card (automatic)", 
    "Electronic check", 
    "Mailed check"
])


# Collecting input data in the correct format
import pandas as pd

data = pd.DataFrame({
    'gender': [0 if gender == "Female" else 1],  # Female: 0, Male: 1
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [1 if partner == "Yes" else 0],  # Yes: 1, No: 0
    'Dependents': [1 if dependents == "Yes" else 0],  # Yes: 1, No: 0
    'tenure': [tenure],
    'PhoneService': [0 if phone_service == "No" else 1],  # No: 0, Yes: 1
    'MultipleLines': [2 if multiple_lines == "Yes" else 0 if multiple_lines == "No" else 1],  # No: 0, Yes: 2, No phone service: 1
    'InternetService': [0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2],  # DSL: 0, Fiber optic: 1, No: 2
    'OnlineSecurity': [2 if online_security == "Yes" else 0 if online_security == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'OnlineBackup': [2 if online_backup == "Yes" else 0 if online_backup == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'DeviceProtection': [2 if device_protection == "Yes" else 0 if device_protection == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'TechSupport': [2 if tech_support == "Yes" else 0 if tech_support == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'StreamingTV': [2 if streaming_tv == "Yes" else 0 if streaming_tv == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'StreamingMovies': [2 if streaming_movies == "Yes" else 0 if streaming_movies == "No" else 1],  # Yes: 2, No: 0, No internet service: 1
    'Contract': [0 if contract == "Month-to-month" else 1 if contract == "One year" else 2],  # Month-to-month: 0, One year: 1, Two year: 2
    'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],  # Yes: 1, No: 0
    'PaymentMethod': [0 if payment_method == "Bank transfer (automatic)" else 1 if payment_method == "Credit card (automatic)" else 2 if payment_method == "Electronic check" else 3],  # 0: Bank transfer, 1: Credit card, 2: Electronic check, 3: Mailed check
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})



# Convert to DataFrame
input_df = pd.DataFrame(data)

# Ensure features match the training set
expected_features = scaler.get_feature_names_out()  # Get the expected feature names
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Preprocess input
processed_input = preprocess_input(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(processed_input)
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    st.success(f"The model predicts: *{result}*")
