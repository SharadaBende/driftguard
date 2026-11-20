import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

with open("models/baseline_metrics.json") as f:
    FEATURES = json.load(f)["features"]

df = pd.read_parquet("data/processed/taxi_sample.parquet")
model = XGBRegressor()
model.load_model("models/baseline.json")
explainer = shap.TreeExplainer(model)

result = {}
for label, mask in [
    ("2019", df["month"].str.startswith("2019")),
    ("2023-10", df["month"] == "2023-10"),
]:
    sample = df[mask].sample(5_000, random_state=42)
    sv = explainer.shap_values(sample[FEATURES])
    result[label] = pd.Series(abs(sv).mean(axis=0), index=FEATURES)

imp = pd.DataFrame(result).round(3).sort_values("2019", ascending=False)
print(imp.to_string())

Path("reports").mkdir(exist_ok=True)
imp.to_csv("reports/shap_importance.csv")
ax = imp.plot(kind="barh", figsize=(8, 5), title="Mean |SHAP| by feature")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/shap_importance.png", dpi=150)
print("Saved reports/shap_importance.csv and .png")