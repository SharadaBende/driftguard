import json

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

with open("models/baseline_metrics.json") as f:
    FEATURES = json.load(f)["features"]
TARGET = "fare_amount"

df = pd.read_parquet("data/processed/taxi_sample.parquet")

train = df[df["month"].isin(["2022-10", "2023-01", "2023-04"])]
test = df[df["month"].isin(["2023-07", "2023-10"])]

champion = XGBRegressor()
champion.load_model("models/baseline.json")

challenger = XGBRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    tree_method="hist", n_jobs=-1, random_state=42,
)
challenger.fit(train[FEATURES], train[TARGET])

champ_mae = mean_absolute_error(test[TARGET], champion.predict(test[FEATURES]))
chall_mae = mean_absolute_error(test[TARGET], challenger.predict(test[FEATURES]))
improvement = (champ_mae - chall_mae) / champ_mae * 100
promote = chall_mae < champ_mae * 0.95  # must be at least 5% better

print(f"Champion MAE:   ${champ_mae:.2f}")
print(f"Challenger MAE: ${chall_mae:.2f}")
print(f"Improvement:    {improvement:.1f}%")
print("Decision:       ", "PROMOTE challenger" if promote else "KEEP champion")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("driftguard")
with mlflow.start_run(run_name="challenger_2022-10_to_2023-04"):
    mlflow.log_param("train_months", "2022-10,2023-01,2023-04")
    mlflow.log_param("test_months", "2023-07,2023-10")
    mlflow.log_metric("champion_mae", champ_mae)
    mlflow.log_metric("challenger_mae", chall_mae)
    mlflow.log_metric("improvement_pct", improvement)
    mlflow.log_param("decision", "promote" if promote else "keep")

if promote:
    challenger.save_model("models/champion.json")
    print("Saved models/champion.json")