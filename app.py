import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and features
try:
    model = joblib.load('churn_predictor.joblib')
    features = joblib.load('features.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model files not found! Please run the 'churn_model.py' script first.")
    st.stop()

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction 📊")
st.markdown("---")

# Helper function for prediction
def predict_churn(input_df):
    prediction = model.predict(input_df)
    churn_proba = model.predict_proba(input_df)[0][1]
    return prediction[0], churn_proba

with st.form("churn_prediction_form"):
    st.header("Customer Details")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    
    with col2:
        st.subheader("Services")
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    
    with col3:
        st.subheader("Billing")
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=20.0, step=0.1)
        total_charges = st.number_input('Total Charges', min_value=0.0, max_value=9000.0, value=500.0, step=0.1)
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'tenure': tenure,
        'Contract': contract,
        'InternetService': internet_service,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'PaymentMethod': payment_method,
        'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'No',
        'MultipleLines': 'No phone service', 'OnlineBackup': 'No internet service',
        'DeviceProtection': 'No internet service', 'StreamingTV': 'No internet service',
        'StreamingMovies': 'No internet service', 'PaperlessBilling': 'No'
    }])
    
    input_df = pd.DataFrame(columns=features)
    input_df = pd.concat([input_df, input_data], ignore_index=True)
    
    prediction, churn_proba = predict_churn(input_df)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Prediction: **This customer is likely to CHURN.** 💔")
        st.write(f"Confidence: **{churn_proba*100:.2f}%**")
    else:
        st.success(f"Prediction: **This customer is likely to STAY.** 🎉")
        st.write(f"Confidence: **{(1 - churn_proba)*100:.2f}%**")

    # Display model explanation using SHAP
    st.subheader("Why This Prediction? (Model Explanation)")
    st.write("The plot below shows which features had the biggest impact on this specific prediction. Red bars indicate features pushing towards 'Churn' (positive class), while blue bars push towards 'Stay' (negative class).")

    # Get the preprocessor and classifier from the pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Get the feature names after preprocessing
    preprocessor_output_features = preprocessor.get_feature_names_out()
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=preprocessor_output_features)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(preprocessed_df)

    # Plot SHAP values
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, preprocessed_df, plot_type="bar", show=False)
    st.pyplot(fig)