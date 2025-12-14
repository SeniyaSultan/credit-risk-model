import pandas as pd
import joblib

def load_model(path: str = "credit_risk_model.pkl"):
    model = joblib.load(path)
    return model

def predict(df: pd.DataFrame, model):
    return model.predict(df)
