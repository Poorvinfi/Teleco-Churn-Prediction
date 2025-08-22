# Telco Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn for a telecom company. Using a dataset of over 7,000 customer records, I developed a machine learning model to identify which customers are likely to terminate their service. The goal is to provide actionable insights to the business, enabling proactive retention strategies.

The project demonstrates a complete data science workflow, from data cleaning and model building to deployment and explainability.

---

## Key Features

* **Data Preprocessing**: Handled missing values, encoded categorical features, and scaled numerical data.
* **Class Imbalance Handling**: Addressed the imbalanced nature of the dataset using **SMOTE** (Synthetic Minority Over-sampling Technique) to improve model performance on the minority class (churning customers).
* **Advanced Modeling**: Trained and optimized an **XGBoost Classifier**, a powerful gradient boosting algorithm known for its high performance on structured data.
* **Model Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** to provide insight into individual predictions. This allows a user to understand *why* a specific customer is predicted to churn.
* **Interactive Deployment**: Deployed the model as an interactive web application using **Streamlit**, making it accessible and easy for business users to test predictions.

---

## Business Insights & Findings

The model identified several key drivers of customer churn:

1.  **Contract Type**: Customers on a **Month-to-month contract** are at the highest risk of churning compared to those with one- or two-year contracts.
2.  **Lack of Service**: The absence of essential services like **Online Security** and **Tech Support** is a strong predictor of churn.
3.  **Tenure**: Customers with a **low tenure** (new customers) are significantly more likely to churn than long-term customers.

These insights can be used by the business to offer targeted promotions, such as upgrading a month-to-month contract to a one-year plan or bundling tech support services.

---

## Technologies Used

* **Python**: The core programming language for the project.
* **Pandas**: For data manipulation and analysis.
* **Scikit-learn**: For data preprocessing and model evaluation.
* **Imbalanced-learn**: To handle class imbalance with SMOTE.
* **XGBoost**: The primary machine learning algorithm.
* **SHAP**: For model explainability.
* **Streamlit**: For creating the interactive web application.
* **Joblib**: To save and load the trained model pipeline.

---

## How to Run the Project Locally

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Poorvinfi/Teleco-Churn-Prediction.git](https://github.com/Poorvinfi/Teleco-Churn-Prediction.git)
    cd Teleco-Churn-Prediction
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the dataset**:
    * Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle.
    * Place it in the project directory.
5.  **Train the model**:
    ```bash
    python churn_model.py
    ```
    This will generate `churn_predictor.joblib` and `features.joblib`.
6.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

---

## Live Demo

You can interact with the live application deployed on Streamlit Cloud here:

**[Paste your Streamlit App URL here]**
