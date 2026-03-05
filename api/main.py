# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

# ✅ Initialize FastAPI
app = FastAPI(title="Credit Card Fraud Detection API")

# ✅ Dynamically build path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# ✅ Load model safely
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

# ✅ Expected features
EXPECTED_FEATURES = 30  # Model trained on 30 features

# ✅ Pydantic schema for single transaction
class Transaction(BaseModel):
    transaction: List[float]

# ✅ Pydantic schema for batch transactions
class BatchTransactions(BaseModel):
    transactions: List[List[float]]

# ✅ Home route
@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

# ✅ Single transaction prediction
@app.post("/predict")
def predict(data: Transaction):
    txn = data.transaction.copy()
    # Trim or pad features
    if len(txn) > EXPECTED_FEATURES:
        txn = txn[:EXPECTED_FEATURES]
    elif len(txn) < EXPECTED_FEATURES:
        txn += [0.0] * (EXPECTED_FEATURES - len(txn))
    arr = np.array(txn).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": int(pred)}

# ✅ Batch transaction prediction
@app.post("/predict_batch")
def predict_batch(data: BatchTransactions):
    predictions = []
    for txn in data.transactions:
        # Trim or pad each transaction
        if len(txn) > EXPECTED_FEATURES:
            txn = txn[:EXPECTED_FEATURES]
        elif len(txn) < EXPECTED_FEATURES:
            txn += [0.0] * (EXPECTED_FEATURES - len(txn))
        arr = np.array(txn).reshape(1, -1)
        pred = model.predict(arr)[0]
        predictions.append(int(pred))
    return {"predictions": predictions}

# ✅ Run with Python directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
