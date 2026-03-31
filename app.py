from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("churn_model.pkl")

columns = joblib.load("columns.pkl")
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
@app.post("/predict")
def predict(data: CustomerData):
    
    input_df = pd.DataFrame([data.dict()])
    
    input_df = input_df.reindex(columns=columns)
    
    
    input_df = input_df.fillna(0)
    
    prob = model.predict_proba(input_df)[0][1]
    
    decision = "High Risk" if prob > 0.3 else "Low Risk"
    
    return {
        "churn_probability": float(prob),
        "decision": decision
    }
