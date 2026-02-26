from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoders
model = joblib.load("fraud_model.pkl")
le_merchant = joblib.load("merchant_encoder.pkl")
le_device = joblib.load("device_encoder.pkl")

# Request body schema
class Transaction(BaseModel):
    amount: float
    merchant_type: str
    device_type: str

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: Transaction):
    try:
        # Convert to dict
        input_dict = data.dict()

        # Encode categorical values
        input_dict["merchant_type"] = le_merchant.transform(
            [input_dict["merchant_type"]]
        )[0]

        input_dict["device_type"] = le_device.transform(
            [input_dict["device_type"]]
        )[0]

        # Convert to DataFrame (VERY IMPORTANT)
        input_df = pd.DataFrame([input_dict])

        # Prediction
        prediction = model.predict(input_df)[0]

        return {
            "fraud_prediction": int(prediction)
        }

    except Exception as e:
        return {"error": str(e)}