# Customer Churn Prediction System

An end-to-end Machine Learning system to predict customer churn using the Telco dataset. This project covers model development, evaluation, explainability, and deployment as a production-ready API.

# Features

- Built multiple ML models (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting)
- Optimized for business goal (high recall for churn detection)
- Threshold tuning for better decision-making
- Feature importance analysis & SHAP explainability
- Production-ready FastAPI deployment
- Handles real-world issues like schema mismatch & missing values

# Problem Statement

Predict whether a customer will churn (leave the service) so that businesses can take proactive retention actions.

# Model Performance

- Logistic Regression (Final Model)
- ROC-AUC: ~0.83
- Recall (Churn class): ~0.76 (after threshold tuning)

# Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- FastAPI
- Uvicorn
- SHAP (Explainability)

# Project Structure

app.py # FastAPI application
churn_model.pkl # Trained ML model
columns.pkl # Feature schema
Telco-Customer-Churn.csv # Dataset
codeCCP.ipynb # Model development
README.md

# Endpoint:


# Example Input:

json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 90,
  "TotalCharges": 180
}

# Example Output:

{
  "churn_probability": 0.82,
  "decision": "High Risk"
}
