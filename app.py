# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:47:03 2024

@author: user
"""

import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load the trained model and training data
model = load_model('diabetes_prediction_model')
train_data = joblib.load('diabetes_train_data.pkl')

# Function to get user input
def get_user_input():
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0)
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    gen_health = st.selectbox('General Health', [1, 2, 3, 4, 5], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x-1])
    income = st.selectbox('Income', [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: ["< $10,000", "$10,000 - $15,000", "$15,000 - $20,000", "$20,000 - $25,000", "$25,000 - $35,000", "$35,000 - $50,000", "$50,000 - $75,000", "≥ $75,000"][x-1])
    ment_health = st.number_input('Mental Health (days)', min_value=0, max_value=30, value=0)
    phys_health = st.number_input('Physical Health (days)', min_value=0, max_value=30, value=0)
    education = st.selectbox('Education Level', [1, 2, 3, 4, 5, 6], format_func=lambda x: ["Never attended school or only kindergarten", "Grades 1 through 8 (Elementary)", "Grades 9 through 11 (Some high school)", "Grade 12 or GED (High school graduate)", "College 1 year to 3 years (Some college or technical school)", "College 4 years or more (College graduate)"][x-1])
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: ["Female", "Male"][x])
    high_chol = st.selectbox('High Cholesterol', [0, 1], format_func=lambda x: ["No", "Yes"][x])
    hvy_alcohol = st.selectbox('Heavy Alcohol Consumption', [0, 1], format_func=lambda x: ["No", "Yes"][x])

    user_data = {
        'BMI': bmi,
        'Age': age,
        'GenHlth': gen_health,
        'Income': income,
        'MentHlth': ment_health,
        'PhysHlth': phys_health,
        'Education': education,
        'Sex': sex,
        'HighChol': high_chol,
        'HvyAlcoholConsump': hvy_alcohol
    }

    return pd.DataFrame(user_data, index=[0])

# Prediction Page
def prediction():
    st.title("Diabetes Prediction")
    user_input = get_user_input()

    if st.button("Predict"):
        prediction_result = predict_model(model, data=user_input)

        if 'prediction_label' in prediction_result.columns and 'prediction_score' in prediction_result.columns:
            predicted_class = int(prediction_result['prediction_label'].iloc[0])
            predicted_prob = prediction_result['prediction_score'].iloc[0]

            st.write("Prediction Result:")
            st.write(f"Predicted Class: {'Diabetic' if predicted_class == 1 else 'Non-Diabetic'}")
            st.write(f"Probability: {predicted_prob:.4f}")

            st.subheader("Prediction Explanation")
            if predicted_class == 1:
                st.write("The model predicts that the patient is likely to have diabetes.")
            else:
                st.write("The model predicts that the patient is unlikely to have diabetes.")

            st.subheader("Probability Explanation")
            st.write("""
                The probability score indicates the model's confidence in its prediction.
                A higher score closer to 1 means the model is more confident that the patient has diabetes,
                while a score closer to 0 means the model is more confident that the patient does not have diabetes.
            """)

            # Risk Factor Analysis using SHAP
            st.subheader("Risk Factor Analysis")
            shap.initjs()

            feature_columns = [col for col in train_data.columns if col != 'Diabetes_binary']
            explainer = shap.Explainer(model.predict, train_data[feature_columns])
            shap_values = explainer(user_input)

            st.write("SHAP Values for Risk Factors:")
            st.write(shap_values.values)

            # Feature Importance Plot
            st.subheader("Feature Importance")
            st.write("""
                The Feature Importance plot shows the most significant factors contributing to the prediction.
                The higher the bar, the more influence that feature has on the model's prediction.
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
            st.pyplot(fig)

            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            st.write("""
                The SHAP Summary plot provides a visualization of the impact of each feature on the model's output.
                Each dot represents a feature's impact on the prediction for a specific instance. 
                The color represents the value of the feature (red for high, blue for low).
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, show=False)
            st.pyplot(fig)

            # SHAP Force Plot
            st.subheader("SHAP Force Plot")
            st.write("""
                The SHAP Force plot illustrates the impact of each feature on the model's prediction for a single instance.
                Features pushing the prediction towards a higher value (indicating diabetes) are shown in red, 
                while those pushing towards a lower value (indicating non-diabetes) are in blue.
            """)
            st_shap(shap.force_plot(shap_values.base_values[0], shap_values.values[0], user_input))

            st.subheader("Detailed SHAP Explanation")
            st.write("""
                SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance.
                The SHAP value of each feature for a given prediction represents how much that feature
                contributes to the difference between the actual prediction and the average prediction for the dataset.
            """)

            for i, feature in enumerate(user_input.columns):
                st.write(f"**{feature}**: SHAP value = {shap_values.values[0][i]:.4f}")

            st.write("""
                - Positive SHAP values indicate that the feature contributes to predicting a higher probability of diabetes.
                - Negative SHAP values indicate that the feature contributes to predicting a lower probability of diabetes.
                - The magnitude of the SHAP value shows the strength of the contribution.
            """)

        else:
            st.error("Error: Prediction did not return 'prediction_label' or 'prediction_score' columns.")

# Helper function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Main function
def main():
    prediction()

if __name__ == "__main__":
    main()
