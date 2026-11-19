import sys
from pathlib import Path

import requests

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
DATA_DIR = Path("data/raw")


def download_month(ym: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"yellow_{ym}.parquet"
    if out.exists():
        print(f"{ym}: already downloaded, skipping")
        return
    print(f"{ym}: downloading...")
    r = requests.get(BASE_URL.format(ym=ym), stream=True, timeout=60)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"{ym}: saved ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for ym in sys.argv[1:]:
        download_month(ym)