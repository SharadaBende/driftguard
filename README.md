# DriftGuard: ML Model Drift Detection and Monitoring

![CI](https://github.com/SharadaBende/driftguard/actions/workflows/ci.yml/badge.svg)

DriftGuard monitors a fare-prediction model after deployment. It trains an XGBoost model on NYC Yellow Taxi trips from 2019, then tracks input drift, target drift and model performance across 20 months of real data (2019 to 2023, about 73 million trips).

## Headline results

- The model's MAE rose from about $1.13 (2019 baseline) to $5.79 in 2023-10, which is 5.1x the baseline.
- In 2023 the model under-predicted fares by up to 28%, while input feature drift stayed low (PSI under 0.07). Monitoring inputs alone would have missed it. Only target drift and performance monitoring caught it.
- In 2020-04 (COVID), input drift was severe (location PSI 0.64) but the money impact was small because trip volume collapsed by 97%.
- Estimated excess prediction error over the 20 sampled months is about $57.2M, and the four 2023 months account for about 88% of it.
- Retraining on recent months cut MAE from $5.44 to $2.65 (51% better) on two unseen months, so the challenger model was promoted.
