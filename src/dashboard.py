import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="DriftGuard", layout="wide")
st.title("DriftGuard: ML Model Drift Monitoring")
st.caption("NYC Yellow Taxi fare model, trained on 2019 data and monitored through 2023")

summary = pd.read_csv("reports/drift_summary.csv")
perf = pd.read_csv("reports/monthly_performance.csv")
scores = pd.read_csv("reports/drift_scores.csv")
impact = pd.read_csv("reports/business_impact.csv")

months = summary["month"].tolist()
month = st.sidebar.select_slider("Month", options=months, value=months[-1])
row = summary[summary["month"] == month].iloc[0]
prow = perf[perf["month"] == month].iloc[0]
irow = impact[impact["month"] == month].iloc[0]

icon = {"green": "🟢", "amber": "🟠", "red": "🔴"}
st.subheader(f"Status for {month}: {icon[row['overall']]} {row['overall'].upper()}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"${prow['mae']:.2f}", f"{row['mae_ratio']:.2f}x baseline", delta_color="off")
c2.metric("Max input PSI", f"{row['max_input_psi']:.3f}", row["top_drift_feature"], delta_color="off")
c3.metric("Target (fare) PSI", f"{row['target_psi']:.3f}", delta_color="off")
c4.metric("Excess error", f"${irow['excess_error_usd'] / 1e6:.1f}M", delta_color="off")

left, right = st.columns(2)
with left:
    fig = px.line(perf, x="month", y="mae", markers=True, title="Model MAE by month")
    fig.add_vline(x=month, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
with right:
    heat = scores.pivot(index="feature", columns="month", values="psi")
    fig2 = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r",
                     zmin=0, zmax=0.5, title="PSI heatmap (feature x month)")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(impact, x="month", y="excess_error_usd", title="Estimated excess prediction error (USD)")
st.plotly_chart(fig3, use_container_width=True)

st.dataframe(summary, use_container_width=True)