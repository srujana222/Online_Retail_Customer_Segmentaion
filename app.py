import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


class Customer(BaseModel):
    age: int
    annual_income: float
    months_active: int
    avg_monthly_spend: float
    purchase_frequency: int
    avg_order_value: float
    discount_usage_rate: float
    return_rate: float
    browsing_time_minutes: float
    support_interactions: int
    payment_method: str
    region: str


@app.post("/predict")
def predict(data: Customer):

    # 🔹 Encoding maps
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
        payment,
        region
    ]])

    
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    return {"prediction": int(prediction[0])}
