import duckdb

query = r"""
SELECT
    regexp_extract(filename, 'yellow_(\d{4}-\d{2})', 1) AS month,
    COUNT(*) AS trips,
    ROUND(median(fare_amount), 2) AS med_fare,
    ROUND(median(trip_distance), 2) AS med_dist,
    ROUND(AVG(tip_amount), 2) AS avg_tip,
    ROUND(AVG(passenger_count), 2) AS avg_pax,
    ROUND(100.0 * AVG(CASE WHEN payment_type = 1 THEN 1 ELSE 0 END), 1) AS pct_card,
    ROUND(median(date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime)), 1) AS med_minutes
FROM read_parquet('data/raw/*.parquet', filename=true, union_by_name=true)
GROUP BY month
ORDER BY month
"""

df = duckdb.sql(query).df()
print(df.to_string(index=False))