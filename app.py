"""Streamlit app for diabetes prediction."""

import joblib
import numpy as np
import streamlit as st


MODEL_PATH = "diabetes_model.joblib"
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    return joblib.load(MODEL_PATH)


def main() -> None:
    """Render the Streamlit UI."""
    st.title("Diabetes Prediction App")
    st.write("Enter patient details to predict diabetes outcome.")

    model = load_model()

    input_values = []
    for feature in FEATURE_NAMES:
        if feature in {"Pregnancies", "Age"}:
            value = st.number_input(feature, min_value=0, max_value=120, step=1)
        else:
            value = st.number_input(feature, min_value=0.0, max_value=500.0, step=0.1)
        input_values.append(value)

    if st.button("Predict"):
        data = np.array([input_values])
        prediction = model.predict(data)[0]
        label = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.subheader(f"Prediction: {label}")


if __name__ == "__main__":
    main()
