from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# -----------------------------
# Create FastAPI App
# -----------------------------

app = FastAPI()

# -----------------------------
# Load Saved Objects
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
pca = joblib.load(BASE_DIR / "pca.pkl")

# -----------------------------
# Request Schema
# -----------------------------

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# -----------------------------
# Home Endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Iris Classification API is running successfully!"
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(data: IrisInput):

    df = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    scaled = scaler.transform(df)

    transformed = pca.transform(scaled)

    prediction = model.predict(transformed)[0]

    return {
        "prediction": prediction
    }