import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('./breast_cancer_model.pkl')

# Function to predict breast cancer


def predict_breast_cancer(features):
    # Convert the features to a NumPy array and reshape it
    features = np.array(features).reshape(1, -1)
    # Make the prediction
    prediction = model.predict(features)
    # Return the prediction
    return prediction[0]

# Streamlit app


def main():
    # Set the title and description
    st.title("Breast Cancer Prediction")
    st.write(
        "Enter the following information to predict the likelihood of breast cancer.")

    # Add some space and set the layout
    st.markdown("---")
    st.markdown(
        '<style>div.row-widget.stNumberInput > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # Create the input fields
    radius_mean = st.number_input(
        "Radius Mean", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    texture_mean = st.number_input(
        "Texture Mean", min_value=0.0, max_value=40.0, value=20.0, step=0.1)
    perimeter_mean = st.number_input(
        "Perimeter Mean", min_value=0.0, max_value=250.0, value=100.0, step=0.1)
    area_mean = st.number_input(
        "Area Mean", min_value=0.0, max_value=2000.0, value=500.0, step=1.0)
    smoothness_mean = st.number_input(
        "Smoothness Mean", min_value=0.0, max_value=0.3, value=0.1, step=0.001)
    compactness_mean = st.number_input(
        "Compactness Mean", min_value=0.0, max_value=0.5, value=0.2, step=0.001)
    concavity_mean = st.number_input(
        "Concavity Mean", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    symmetry_mean = st.number_input(
        "Symmetry Mean", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean", min_value=0.0, max_value=0.2, value=0.1, step=0.001)

    # Create a feature array from the user input
    features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, symmetry_mean, fractal_dimension_mean]

    # Create a button to predict breast cancer
    if st.button("Predict"):
        # Call the predict_breast_cancer function with the user input features
        result = predict_breast_cancer(features)
        # Display the prediction
        if result == 0:
            st.write("Good news! The tumor is likely to be benign (non-cancerous).")
        else:
            st.write("Warning! The tumor is likely to be malignant (cancerous).")


# Run the app
if __name__ == '__main__':
    main()
