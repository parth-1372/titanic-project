import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# This file 'titanic_model.joblib' MUST be in the same folder
try:
    model = joblib.load('titanic_model.joblib')
except FileNotFoundError:
    st.error("Model file 'titanic_model.joblib' not found. Make sure it's in the same directory.")
    st.stop()

# Set page title
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# App title
st.title("🚢 Titanic Survival Prediction")
st.markdown("This app predicts whether a passenger would have survived the Titanic disaster.")

# --- Input Fields for User ---
# We use columns to organize the layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Info")
    # pclass (Passenger Class)
    pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
    
    # sex
    sex = st.radio("Sex:", ["male", "female"])
    
    # age
    age = st.slider("Age:", 0.0, 80.0, 25.0, 0.5)

with col2:
    st.subheader("Travel Info")
    # sibsp (Siblings/Spouses Aboard)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
    
    # parch (Parents/Children Aboard)
    parch = st.number_input("Parents/Children Aboard (Parch):", 0, 6, 0)
    
    # fare
    fare = st.number_input("Fare (in $):", 0.0, 600.0, 30.0, 1.0)
    
    # embarked
    embarked = st.selectbox("Port of Embarkation:", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")

# --- Prediction Logic ---
if st.button("Predict Survival", help="Click to see the prediction"):
    
    # Create a DataFrame from the inputs
    # The column names MUST match the ones used during training
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked]
    })
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        # Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"**LIKELY TO SURVIVE** (Probability: {probability[1]*100:.2f}%)")
        else:
            st.error(f"**NOT LIKELY TO SURVIVE** (Probability: {probability[0]*100:.2f}%)")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Project by [Your Name Here]")