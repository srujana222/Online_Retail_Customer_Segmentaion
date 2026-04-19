from fastapi import FastAPI
from pydantic import BaseModel
import pickle,joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
import joblib
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class CustomerData(BaseModel):
    age: float
    annual_income: float
    months_active: float
    avg_monthly_spend: float
    purchase_frequency: float
    avg_order_value: float
    discount_usage_rate: float
    return_rate: float
    browsing_time_minutes: float
    support_interactions: float
    payment_method:str
    region:str      

@app.get("/")
def home():
    return {"message": "Customer Classification API is running"}

@app.post("/predict")
def predict(data: CustomerData):
     payment_map = {
        "Credit Card": 0,
        "Debit Card": 1,
        "UPI": 2,
        "Cash": 3
    }

    region_map = {
        "North": 0,
        "South": 1,
        "East": 2,
        "West": 3
    }
    payment = payment_map.get(data.payment_method, 0)
    region = region_map.get(data.region, 0)

    
    input_data = np.array([[
        data.age,
        data.annual_income,
        data.months_active,
        data.avg_monthly_spend,
        data.purchase_frequency,
        data.avg_order_value,
        data.discount_usage_rate,
        data.return_rate,
        data.browsing_time_minutes,
        data.support_interactions,
        data.payment_method,
        data.region       
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)

    return {
        "prediction": result[0]
    }
