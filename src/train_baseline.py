import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURES = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "pickup_hour",
    "pickup_dow",
]
TARGET = "fare_amount"

df = pd.read_parquet("data/processed/taxi_sample.parquet")
base = df[df["month"].str.startswith("2019")]
print(f"Baseline training data (2019): {len(base):,} rows")

X_train, X_val, y_train, y_val = train_test_split(
    base[FEATURES], base[TARGET], test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_val)
mae = mean_absolute_error(y_val, pred)
rmse = mean_squared_error(y_val, pred) ** 0.5
print(f"Validation MAE:  ${mae:.2f}")
print(f"Validation RMSE: ${rmse:.2f}")

Path("models").mkdir(exist_ok=True)
model.save_model("models/baseline.json")
with open("models/baseline_metrics.json", "w") as f:
    json.dump({"mae": mae, "rmse": rmse, "features": FEATURES}, f, indent=2)
print("Saved models/baseline.json")