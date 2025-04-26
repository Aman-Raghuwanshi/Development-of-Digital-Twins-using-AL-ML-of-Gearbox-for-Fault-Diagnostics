import streamlit as st
import pandas as pd
import joblib

# Load models and preprocessing objects
svm_model = joblib.load('svm_model.pkl')
lr_model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Streamlit UI - Title and description
st.title("Faulty Gear Prediction App")
st.write("This app predicts if a gear system is faulty based on input features.")

# Sidebar - Model selection
model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "SVM"])

# Input fields for user to enter data
st.header("Enter Gear Analysis Data")

def user_input_features():
    feature1 = st.number_input("No. of Teeth on Gear", min_value=0)
    feature2 = st.number_input("No. of Teeth on Pinion", min_value=0)
    feature3 = st.number_input("Average Gear Ratio", min_value=0.0)
    feature4 = st.number_input("Total Deformation", min_value=0.0)
    feature5 = st.number_input("Equivalent Stress", min_value=0.0)
    feature6 = st.number_input("Maximum Principal Elastic Strain", min_value=0.0)
    feature7 = st.number_input("Chip Length", min_value=0.0)

    # Create a DataFrame with the user input
    data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]],
                        columns=['No. of Teeth on Gear', 'No. of Teeth on Pinion', 
                                 'Average Gear Ratio', 'Total Deformation',
                                 'Equivalent Stress', 'Maximum Principal Elastic Strain', 'Chip Length'])
    return data

input_df = user_input_features()

# Show the user input as a table
st.write("User Input Data:")
st.write(input_df)

# When the user clicks the 'Predict' button, make the prediction
if st.button("Predict"):
    # Preprocess the input data
    imputed_input = imputer.transform(input_df)  # Impute missing values
    scaled_input = scaler.transform(imputed_input)  # Scale the input data

    # Make prediction based on selected model
    if model_choice == "Logistic Regression":
        prediction = lr_model.predict(scaled_input)
    else:
        prediction = svm_model.predict(scaled_input)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Prediction: The gear system is FAULTY!")
    else:
        st.success("Prediction: The gear system is NOT FAULTY!")
