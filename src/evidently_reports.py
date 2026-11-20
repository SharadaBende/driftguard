from pathlib import Path

import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

COLS = [
    "trip_distance", "passenger_count", "PULocationID", "DOLocationID",
    "RatecodeID", "pickup_hour", "pickup_dow", "fare_amount",
]

df = pd.read_parquet("data/processed/taxi_sample.parquet")
reference = df[df["month"].str.startswith("2019")][COLS].sample(100_000, random_state=42)

Path("reports").mkdir(exist_ok=True)
for month in ["2020-04", "2023-10"]:
    current = df[df["month"] == month][COLS]
    current = current.sample(min(100_000, len(current)), random_state=42)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out = f"reports/evidently_{month}.html"
    report.save_html(out)
    print("Saved", out)