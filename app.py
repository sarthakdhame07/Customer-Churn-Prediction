import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl

# Load the model
try:
    model = pkl.load(open('MIMPL.pkl', 'rb'))
    scaler= pkl.load(open('scaler.pkl','rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'MIMPL.pkl' is in the correct directory.")
    st.stop()

st.title("Customer Churn Prediction")

st.markdown("""
### Predict whether the customer is likely to churn or not
Enter the correct details of the customer and click *Predict Churn* button to display the result
""")

# Input fields
age = st.number_input('Enter your age', min_value=10, max_value=80, value=25)
total_spend = st.number_input('Enter Spend amount', min_value=0, max_value=100000, value=1000)
item_purchased = st.number_input('Enter the no of purchased items', min_value=0, max_value=500, value=5)
avg_rating = st.slider('Average Rating', 1.0, 5.0, 3.5)
discount_applied = st.number_input('Enter Discount Applied %', min_value=0, max_value=100, value=10)
days_since_last_purchase = st.number_input('Days since last Purchase', min_value=0, max_value=365, value=30)

# Create input DataFrame
input_data = pd.DataFrame([[age, total_spend, item_purchased, avg_rating, discount_applied, days_since_last_purchase]],
                          columns=['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase'])

# Debugging: Print input data
st.write("Input Data:", input_data)

# Predict churn
if st.button('Predict Churn'):
    input_scaled=scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    churn_probability = model.predict_proba(input_scaled)[0][1] * 100

    # Debugging: Print prediction and probability
    st.write("Prediction:", prediction)
    st.write("Churn Probability:", churn_probability)

    if prediction == 1:
        st.error(f'The customer has a **{churn_probability:.2f}%** chance of churning. Suggest giving a **{discount_applied}**% discount.')
    else:
        st.success(f'This customer is **not likely to churn**. **{churn_probability:.2f}%**.')