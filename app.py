import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# Load the pre-trained model
model = joblib.load('breast_cancer_model.pkl')

# Create an instance of RobustScaler
scaler = RobustScaler()

# Define dictionaries for labels
label_dict = {0: 'Benign', 1: 'Malignant'}

df = pd.read_csv("./datasets/data.csv")
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
le = LabelEncoder()
# label data between 0 and 1
df['diagnosis'] = le.fit_transform(df['diagnosis'])

x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], df['diagnosis'], test_size=0.2, random_state=42)
scale = RobustScaler()
scale.fit_transform(x_train)
# Function to predict breast cancer


def predict_breast_cancer(input_data):
    # Convert the features to a NumPy array and reshape it
    input_data = np.asarray(input_data)
    # Make the prediction
    input_data = input_data.reshape(1, -1)
    input_data = scale.transform(input_data)
    prediction = model.predict(input_data)

    # Return the prediction
    return label_dict[prediction[0]]


# Streamlit app
def main():
    # Set the title and description
    st.title("Breast Cancer Prediction")
    st.write(
        "Enter the following information to predict the likelihood of breast cancer.")
    st.write("**Note: This is a sensitive prediction model, ensure that you enter the data accurately and responsibly.**")

    # Add some space and set the layout
    st.markdown("---")

    # Create the input fields
    features = {}
    features['radius_mean'] = st.number_input(
        "Radius Mean", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    features['texture_mean'] = st.number_input(
        "Texture Mean", min_value=0.0, max_value=40.0, value=20.0, step=0.1)
    features['perimeter_mean'] = st.number_input(
        "Perimeter Mean", min_value=0.0, max_value=250.0, value=100.0, step=0.1)
    # Add more input fields for the other features

    # Create a button to predict breast cancer
    if st.button("Predict"):
        # Call the predict_breast_cancer function with the input features
        result = predict_breast_cancer(list(features.values()))
        # Display the prediction
        st.write("Prediction:", result)


# Run the app
if __name__ == '__main__':
    main()
