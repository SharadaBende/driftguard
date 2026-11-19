import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

with open("models/baseline_metrics.json") as f:
    FEATURES = json.load(f)["features"]
TARGET = "fare_amount"

df = pd.read_parquet("data/processed/taxi_sample.parquet")

model = XGBRegressor()
model.load_model("models/baseline.json")

rows = []
for month, g in df.groupby("month"):
    pred = model.predict(g[FEATURES])
    rows.append(
        {
            "month": month,
            "rows": len(g),
            "mae": mean_absolute_error(g[TARGET], pred),
            "rmse": mean_squared_error(g[TARGET], pred) ** 0.5,
            "mean_actual": g[TARGET].mean(),
            "mean_pred": pred.mean(),
            "bias": (pred - g[TARGET]).mean(),
        }
    )

result = pd.DataFrame(rows).round(2)
print(result.to_string(index=False))

Path("reports").mkdir(exist_ok=True)
result.to_csv("reports/monthly_performance.csv", index=False)
print("\nSaved reports/monthly_performance.csv")