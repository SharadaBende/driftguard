import duckdb

query = r"""
SELECT
    regexp_extract(filename, 'yellow_(\d{4}-\d{2})', 1) AS month,
    COUNT(*) AS trips
FROM read_parquet('data/raw/*.parquet', filename=true, union_by_name=true)
GROUP BY month
ORDER BY month
"""

df = duckdb.sql(query).df()
print(df.to_string(index=False))
print("\nTotal trips:", f"{df['trips'].sum():,}")