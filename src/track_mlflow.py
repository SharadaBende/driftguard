import json

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

with open("models/baseline_metrics.json") as f:
    FEATURES = json.load(f)["features"]
TARGET = "fare_amount"

df = pd.read_parquet("data/processed/taxi_sample.parquet")
model = XGBRegressor()
model.load_model("models/baseline.json")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("driftguard")

with mlflow.start_run(run_name="baseline_2019"):
    mlflow.log_params(model.get_params())
    mlflow.log_param("train_period", "2019")
    mlflow.log_param("features", ",".join(FEATURES))
    for month, g in df.groupby("month"):
        pred = model.predict(g[FEATURES])
        step = int(month.replace("-", ""))
        mlflow.log_metric("mae", mean_absolute_error(g[TARGET], pred), step=step)
        mlflow.log_metric("rmse", mean_squared_error(g[TARGET], pred) ** 0.5, step=step)
    mlflow.log_artifact("models/baseline.json")
    mlflow.log_artifact("reports/drift_summary.csv")
    print("Logged run to MLflow")