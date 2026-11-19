from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

NUMERIC = ["trip_distance", "pickup_hour", "fare_amount"]
CATEGORICAL = ["passenger_count", "RatecodeID", "pickup_dow", "PULocationID", "DOLocationID"]
EPS = 1e-6


def psi_from_props(r, c):
    r = np.clip(r, EPS, None)
    c = np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return psi_from_props(r, c)


def categorical_counts(ref, cur):
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    rc = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
    cc = cur.value_counts().reindex(cats, fill_value=0).to_numpy()
    return rc, cc


def psi_categorical(ref, cur):
    rc, cc = categorical_counts(ref, cur)
    return psi_from_props(rc / rc.sum(), cc / cc.sum())


def cramers_v(ref, cur):
    rc, cc = categorical_counts(ref, cur)
    table = np.array([rc, cc])
    chi2, _, _, _ = chi2_contingency(table)
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, month: str) -> list[dict]:
    rows = []
    for col in NUMERIC:
        rows.append({
            "month": month, "feature": col, "type": "numeric",
            "psi": psi_numeric(reference[col].to_numpy(), current[col].to_numpy()),
            "ks_stat": ks_2samp(reference[col], current[col]).statistic,
            "wasserstein": wasserstein_distance(reference[col], current[col]),
            "cramers_v": np.nan,
        })
    for col in CATEGORICAL:
        rows.append({
            "month": month, "feature": col, "type": "categorical",
            "psi": psi_categorical(reference[col], current[col]),
            "ks_stat": np.nan, "wasserstein": np.nan,
            "cramers_v": cramers_v(reference[col], current[col]),
        })
    return rows


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/taxi_sample.parquet")
    reference = df[df["month"].str.startswith("2019")]

    all_rows = []
    for month, g in df.groupby("month"):
        all_rows.extend(compute_drift(reference, g, month))
        print(f"{month} done")

    result = pd.DataFrame(all_rows)
    Path("reports").mkdir(exist_ok=True)
    result.to_csv("reports/drift_scores.csv", index=False)

    pivot = result.pivot(index="month", columns="feature", values="psi").round(3)
    print("\nPSI by month and feature:")
    print(pivot.to_string())