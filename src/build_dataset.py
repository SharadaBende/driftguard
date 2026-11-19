from pathlib import Path

import duckdb
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = 200_000

frames = []
for f in sorted(RAW.glob("yellow_*.parquet")):
    month = f.stem.replace("yellow_", "")
    query = f"""
    SELECT
        '{month}' AS month,
        fare_amount,
        trip_distance,
        passenger_count,
        PULocationID,
        DOLocationID,
        RatecodeID,
        payment_type,
        hour(tpep_pickup_datetime) AS pickup_hour,
        dayofweek(tpep_pickup_datetime) AS pickup_dow,
        date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime) AS duration_min
    FROM read_parquet('{f.as_posix()}')
    WHERE strftime(tpep_pickup_datetime, '%Y-%m') = '{month}'
      AND fare_amount BETWEEN 2.5 AND 200
      AND trip_distance BETWEEN 0.1 AND 100
      AND passenger_count BETWEEN 1 AND 6
      AND date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime) BETWEEN 1 AND 180
    USING SAMPLE {SAMPLE} ROWS
    """
    df = duckdb.sql(query).df()
    print(f"{month}: {len(df):,} rows kept")
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full.to_parquet(OUT / "taxi_sample.parquet", index=False)
print(f"\nSaved {len(full):,} rows to {OUT / 'taxi_sample.parquet'}")
