import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

MODEL_PATH = Path("models/champion.json")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/baseline.json")

with open("models/baseline_metrics.json") as f:
    FEATURES = json.load(f)["features"]

model = XGBRegressor()
model.load_model(str(MODEL_PATH))

app = FastAPI(title="DriftGuard API", version="1.0")


class Trip(BaseModel):
    trip_distance: float = Field(gt=0, le=100)
    passenger_count: int = Field(ge=1, le=6)
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    RatecodeID: int = Field(ge=1, le=6)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_dow: int = Field(ge=0, le=6)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict(trip: Trip):
    X = pd.DataFrame([trip.model_dump()])[FEATURES]
    fare = float(model.predict(X)[0])
    return {"predicted_fare_usd": round(fare, 2), "model": MODEL_PATH.name}


@app.get("/drift-report")
def drift_report(month: str | None = None):
    path = Path("reports/drift_summary.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="drift_summary.csv not found")
    df = pd.read_csv(path)
    if month:
        df = df[df["month"] == month]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {month}")
    return df.to_dict(orient="records")