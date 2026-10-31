from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from antifraud.predict import predict_transaction

class Transaction(BaseModel):
    features: list[float] = Field(max_length=165, min_length=165)


app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = transaction.features

    X = pd.DataFrame([features])

    y_pred = predict_transaction(X)
    return {"features": float(y_pred[0])}