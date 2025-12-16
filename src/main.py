from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# Load your trained model from MLflow
model_uri = "runs:/<RUN_ID>/model"  # Replace <RUN_ID> with your MLflow run ID
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI(title="Credit Risk Prediction API")

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    ProductCategory: str
    ChannelId: str
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int

@app.post("/predict")
def predict_risk(data: CustomerData):
    # Convert incoming data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict risk probability
    risk_prob = model.predict_proba(df)[:, 1][0]
    return {"risk_probability": risk_prob}
