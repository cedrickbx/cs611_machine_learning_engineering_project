# read_parquet.py
import pandas as pd
from pathlib import Path
import sys

# Resolve paths relative to this file's location (not the terminal CWD)
ROOT = Path(__file__).resolve().parent

combined_parquet = ROOT / "datamart/gold/gold_combined_historical.parquet/snapshot_date=2023-01-01"
flight_parquet   = ROOT / "datamart/gold/flight/flight_gold_oot_2025_01_01.parquet/snapshot_date=2025-01-01"

def check_exists(p: Path):
    if not p.exists():
        print(f"❌ Path not found: {p}")
        sys.exit(1)

def read_parquet_fast(path: Path):
    # Prefer PyArrow engine; it’s fastest and supports directories
    return pd.read_parquet(path, engine="pyarrow", dtype_backend="pyarrow")

def main():
    check_exists(combined_parquet)
    check_exists(flight_parquet)

    df_combined = read_parquet_fast(combined_parquet)
    df_flight   = read_parquet_fast(flight_parquet)

    print("\nCombined DataFrame Info:")
    print(df_combined.info())
    print("\nShape:", df_combined.shape)

    print("\nFlight DataFrame Info:")
    print(df_flight.info())
    print("\nShape:", df_flight.shape)

    print("\nFirst 5 rows of Combined DataFrame:")
    print(df_combined.head())

    print("\nFirst 5 rows of Flight DataFrame:")
    print(df_flight.head())

if __name__ == "__main__":
    main()
