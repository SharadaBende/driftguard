from pathlib import Path

import pandas as pd

PSI_AMBER, PSI_RED = 0.10, 0.25
MAE_AMBER, MAE_RED = 1.25, 1.75  # multiples of baseline MAE


def psi_level(psi: float) -> str:
    if psi >= PSI_RED:
        return "red"
    if psi >= PSI_AMBER:
        return "amber"
    return "green"


def perf_level(ratio: float) -> str:
    if ratio >= MAE_RED:
        return "red"
    if ratio >= MAE_AMBER:
        return "amber"
    return "green"


def worst(levels) -> str:
    order = {"green": 0, "amber": 1, "red": 2}
    return max(levels, key=lambda x: order[x])


if __name__ == "__main__":
    drift = pd.read_csv("reports/drift_scores.csv")
    perf = pd.read_csv("reports/monthly_performance.csv")

    baseline_mae = perf[perf["month"].str.startswith("2019")]["mae"].mean()
    inputs = drift[drift["feature"] != "fare_amount"]

    rows = []
    for month, g in inputs.groupby("month"):
        top = g.loc[g["psi"].idxmax()]
        target_psi = drift[(drift["month"] == month) & (drift["feature"] == "fare_amount")]["psi"].iloc[0]
        mae = perf.loc[perf["month"] == month, "mae"].iloc[0]
        ratio = mae / baseline_mae
        input_lvl = psi_level(top["psi"])
        target_lvl = psi_level(target_psi)
        perf_lvl = perf_level(ratio)
        rows.append({
            "month": month,
            "top_drift_feature": top["feature"],
            "max_input_psi": round(top["psi"], 3),
            "target_psi": round(target_psi, 3),
            "mae_ratio": round(ratio, 2),
            "input_level": input_lvl,
            "target_level": target_lvl,
            "perf_level": perf_lvl,
            "overall": worst([input_lvl, target_lvl, perf_lvl]),
        })

    summary = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    summary.to_csv("reports/drift_summary.csv", index=False)
    print(summary.to_string(index=False))