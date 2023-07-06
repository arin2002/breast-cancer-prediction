import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('breast_cancer_model.pkl')

# Define dictionaries for labels
label_dict = {0: 'Benign', 1: 'Malignant'}

# Function to predict breast cancer


def predict_breast_cancer(features):
    # Convert the features to a NumPy array and reshape it
    features = np.array(features).reshape(1, -1)
    # Make the prediction
    prediction = model.predict(features)
    # Return the prediction
    return label_dict[prediction[0]]

# Streamlit app


def main():
    # Set the title and description
    st.title("Breast Cancer Prediction")
    st.write(
        "Enter the following information to predict the likelihood of breast cancer.")

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
    features['area_mean'] = st.number_input(
        "Area Mean", min_value=0.0, max_value=2000.0, value=500.0, step=1.0)
    features['smoothness_mean'] = st.number_input(
        "Smoothness Mean", min_value=0.0, max_value=0.3, value=0.1, step=0.001)
    features['compactness_mean'] = st.number_input(
        "Compactness Mean", min_value=0.0, max_value=0.5, value=0.2, step=0.001)
    features['concavity_mean'] = st.number_input(
        "Concavity Mean", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    features['concave_points_mean'] = st.number_input(
        "Concave Points Mean", min_value=0.0, max_value=0.2, value=0.1, step=0.001)
    features['symmetry_mean'] = st.number_input(
        "Symmetry Mean", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['fractal_dimension_mean'] = st.number_input(
        "Fractal Dimension Mean", min_value=0.0, max_value=0.2, value=0.1, step=0.001)
    features['radius_se'] = st.number_input(
        "Radius SE", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    features['texture_se'] = st.number_input(
        "Texture SE", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    features['perimeter_se'] = st.number_input(
        "Perimeter SE", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    features['area_se'] = st.number_input(
        "Area SE", min_value=0.0, max_value=800.0, value=300.0, step=1.0)
    features['smoothness_se'] = st.number_input(
        "Smoothness SE", min_value=0.0, max_value=0.2, value=0.1, step=0.001)
    features['compactness_se'] = st.number_input(
        "Compactness SE", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['concavity_se'] = st.number_input(
        "Concavity SE", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    features['concave_points_se'] = st.number_input(
        "Concave Points SE", min_value=0.0, max_value=0.2, value=0.1, step=0.001)
    features['symmetry_se'] = st.number_input(
        "Symmetry SE", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['fractal_dimension_se'] = st.number_input(
        "Fractal Dimension SE", min_value=0.0, max_value=0.1, value=0.05, step=0.001)
    features['radius_worst'] = st.number_input(
        "Radius Worst", min_value=0.0, max_value=40.0, value=20.0, step=0.1)
    features['texture_worst'] = st.number_input(
        "Texture Worst", min_value=0.0, max_value=60.0, value=30.0, step=0.1)
    features['perimeter_worst'] = st.number_input(
        "Perimeter Worst", min_value=0.0, max_value=300.0, value=150.0, step=0.1)
    features['area_worst'] = st.number_input(
        "Area Worst", min_value=0.0, max_value=4000.0, value=1000.0, step=1.0)
    features['smoothness_worst'] = st.number_input(
        "Smoothness Worst", min_value=0.0, max_value=0.5, value=0.2, step=0.001)
    features['compactness_worst'] = st.number_input(
        "Compactness Worst", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['concavity_worst'] = st.number_input(
        "Concavity Worst", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['concave_points_worst'] = st.number_input(
        "Concave Points Worst", min_value=0.0, max_value=0.5, value=0.2, step=0.001)
    features['symmetry_worst'] = st.number_input(
        "Symmetry Worst", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    features['fractal_dimension_worst'] = st.number_input(
        "Fractal Dimension Worst", min_value=0.0, max_value=1.0, value=0.5, step=0.001)

    # Create a button to predict breast cancer
    if st.button("Predict"):
        # Call the predict_breast_cancer function with the user input features```
        result = predict_breast_cancer(list(features.values()))
        # Display the prediction
        st.write("Prediction:", result)


# Run the app
if __name__ == '__main__':
    main()
