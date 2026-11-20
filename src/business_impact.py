from pathlib import Path

import duckdb
import pandas as pd

perf = pd.read_csv("reports/monthly_performance.csv")

counts = duckdb.sql(r"""
SELECT
    regexp_extract(filename, 'yellow_(\d{4}-\d{2})', 1) AS month,
    COUNT(*) AS total_trips
FROM read_parquet('data/raw/*.parquet', filename=true, union_by_name=true)
GROUP BY month
""").df()

df = perf.merge(counts, on="month")
baseline_mae = df[df["month"].str.startswith("2019")]["mae"].mean()

df["excess_mae"] = (df["mae"] - baseline_mae).clip(lower=0)
df["monthly_error_usd"] = df["mae"] * df["total_trips"]
df["excess_error_usd"] = df["excess_mae"] * df["total_trips"]
df["pct_fare_underpredicted"] = (df["bias"] / df["mean_actual"] * 100).round(1)

out = df[[
    "month", "total_trips", "mae", "excess_mae",
    "monthly_error_usd", "excess_error_usd", "pct_fare_underpredicted",
]].round(2)

pd.options.display.float_format = "{:,.2f}".format
print(out.to_string(index=False))

total_excess = out["excess_error_usd"].sum()
print(f"\nBaseline MAE (2019): ${baseline_mae:.2f}")
print(f"Total excess error across the 20 sampled months: ${total_excess:,.0f}")

Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/business_impact.csv", index=False)
print("Saved reports/business_impact.csv")