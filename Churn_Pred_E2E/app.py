import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle

def get_data(fro_m):
    with open(fro_m, 'rb') as file:
        return pickle.load(file)

gender = get_data('tran_gen.pkl')
geography = get_data('tran_Geo.pkl')

# Load the trained model
import tensorflow as tf # type: ignore

model = tf.keras.models.load_model('Churn_Pred_E2E\model.keras')


# UI and Input Data
st.title('Customer Churn Prediction')
col1, col2 = st.columns(2)

with col1:
    gender_choice = st.selectbox(
        "Gender",
        options=gender['Gender'].unique()
    )

    age = st.slider("Age", 18, 92)
    tenure = st.slider("Tenure", 0, 10)
    number_of_products = st.slider("Number of Products", 1, 4)
    is_active_member = st.selectbox("Is Active Member", [0, 1])

with col2:
    geography_choice = st.selectbox(
        "Geography", 
        options=geography.columns
    )

    estimated_salary = st.number_input("Estimated Salary")
    balance = st.number_input("Balance")
    credit_score = st.number_input("Credit Score")
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_choice],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

transform_geo = pd.DataFrame(0, index=[0], columns=geography.columns)
transform_geo[geography_choice]=1

input_data = pd.concat([input_data.reset_index(drop=True), transform_geo], axis=1)

input_data_scaled = StandardScaler().fit_transform(input_data)

input_data_scaled = pd.DataFrame(input_data_scaled)

prediction = model.predict(input_data)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba : 0.2f}')

if prediction_proba > 0.5:
    st.success('The customer is likely to churn.')
else:
    st.warning('The customer is not likely to churn.')
