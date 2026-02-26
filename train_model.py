import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Training started...")

# Load CSV (NOT Excel)
df = pd.read_csv("fraud_detection.xls")

# Drop ID column
X = df.drop(["transaction_id", "label"], axis=1)
y = df["label"]

# Encode categorical columns
le_merchant = LabelEncoder()
le_device = LabelEncoder()

X["merchant_type"] = le_merchant.fit_transform(X["merchant_type"])
X["device_type"] = le_device.fit_transform(X["device_type"])

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save files in ROOT folder
joblib.dump(model, "fraud_model.pkl")
joblib.dump(le_merchant, "merchant_encoder.pkl")
joblib.dump(le_device, "device_encoder.pkl")

print("Model and encoders saved successfully!")