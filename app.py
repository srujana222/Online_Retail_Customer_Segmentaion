from fastapi import FastAPI
from pydantic import BaseModel
import pickle,joblib
import numpy as np

# Initialize app
app = FastAPI()

# Load model, scaler, encoder
model = joblib.load("model.pkl")
import joblib
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🔷 Define Input Schema (IMPORTANT)
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

# Home route
@app.get("/")
def home():
    return {"message": "Customer Classification API is running"}

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):
    
    # Convert input to array
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

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert back to label
    result = label_encoder.inverse_transform(prediction)

    return {
        "prediction": result[0]
    }