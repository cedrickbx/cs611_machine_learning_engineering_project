# Import required libraries
import pandas as pd
from pathlib import Path

# Define paths to parquet files
combined_parquet = "datamart/gold/combined/test_gold_combined_oot.parquet/snapshot_date=2025-01-15"
flight_parquet = "datamart/gold/flight/test_flight_gold_oot.parquet/snapshot_date=2025-01-15"

# Read parquet files into pandas DataFrames
df_combined = pd.read_parquet(combined_parquet)
df_flight = pd.read_parquet(flight_parquet)

# Display basic information about the DataFrames
print("\nCombined DataFrame Info:")
print(df_combined.info())
print("\nShape:", df_combined.shape)

print("\nFlight DataFrame Info:")
print(df_flight.info())
print("\nShape:", df_flight.shape)

# Display first few rows of each DataFrame
print("\nFirst 5 rows of Combined DataFrame:")
print(df_combined.head())

print("\nFirst 5 rows of Flight DataFrame:")
print(df_flight.head())