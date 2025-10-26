import os, argparse, time
from datetime import datetime
from typing import Iterable

from pyspark.sql import SparkSession

from utils.data_processing_weather_bronze import process_bronze_weather_history
from utils.download_weather_data import download_noaa_isd_data

# Use downloader + constants
try:
    from utils.download_weather_data import (
        download_noaa_isd_data,
        target_stations,        # dict: airport_code -> station_id
        LOCAL_BASE_DIR          # "data/weather_history"
    )
except Exception:
    # Fallbacks if import path differs; adjust if needed
    target_stations = {"KJFK": "74486094789", "KLGA": "72503014732", "KEWR": "72502014734"}
    LOCAL_BASE_DIR = "../data/weather_history"
    def download_noaa_isd_data(start_year=2023, end_year=2024):
        raise RuntimeError("download_noaa_isd_data not found. Ensure utils.download_weather_data is importable.")


def _year_range(start_year: int, end_year: int) -> Iterable[int]:
    return range(start_year, end_year + 1)


def _is_weather_data_present(start_year=2023, end_year=2025) -> bool:
    """
    Returns True if all expected files exist under LOCAL_BASE_DIR:
      ../data/weather_history/<station_id>/<year>.csv
    """
    years = list(_year_range(start_year, end_year))
    if not os.path.isdir(LOCAL_BASE_DIR):
        return False

    # Quick existence check
    for _, station_id in target_stations.items():
        station_dir = os.path.join(LOCAL_BASE_DIR, station_id)
        for yr in years:
            if not os.path.isfile(os.path.join(station_dir, f"{yr}.csv")):
                return False
    return True


def main(snapshotdate=None, download_data=False, oot_start=None, oot_end=None, no_hist=False):
    start = time.time()
    print("\n" + "="*80)
    print("WEATHER HISTORY → BRONZE (Standalone)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode:", "OOT (daily)" if snapshotdate else "Historical (2023–2024)")
    print("="*80 + "\n")

    # --- Data presence check + optional download
    if snapshotdate:
        # For OOT, we still rely on raw history folder layout; if absent, offer to download historical set (2023–2024)
        need_hist = not _is_weather_data_present(2023, 2025)
        if need_hist:
            if download_data:
                print("Raw weather files missing. Downloading historical 2023–2025…")
                download_noaa_isd_data(2023, 2025)
                print("Download complete.")
            else:
                print("Raw weather files missing under ../data/weather_history/.")
                print("Re-run with --download-data to fetch 2023–2025 now, or place files manually.")
                return
    else:
        # Historical run requires 2023–2024 to be present; download if allowed
        if not _is_weather_data_present(2023, 2025):
            if download_data:
                print("Downloading weather historical data (2023–2025)…")
                download_noaa_isd_data(2023, 2025)
                print("Download complete.")
            else:
                print("Historical data not found under ../data/weather_history/.")
                print("Re-run with --download-data to fetch, or place files manually.")
                return

    # --- Spark session
    spark = (SparkSession.builder
             .appName("WeatherHistoryBronze")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "24")
             .config("spark.driver.memory", "4g")
             .config("spark.executor.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    try:
        raw_root = "../data/weather_history"
        out_root = "../datamart/bronze/weather_history"

        if snapshotdate:
            process_bronze_weather_history(
                spark=spark,
                raw_root_dir=raw_root,
                output_root=out_root,
                oot_date_list=[snapshotdate],         # <-- single day only
                write_historical=not no_hist
            )
        elif oot_start and oot_end:
            process_bronze_weather_history(
                spark=spark,
                raw_root_dir=raw_root,
                output_root=out_root,
                oot_start=oot_start,
                oot_end=oot_end,
                write_historical=not no_hist
            )
        else:
            # Historical parquet
            process_bronze_weather_history(
                spark=spark,
                raw_root_dir=raw_root,
                output_root=out_root,
                write_historical=True
            )

            elapsed = (time.time() - start) / 60.0
            print(f"\n✓ Done in {elapsed:.1f} minutes")

    finally:
        print("\nStopping Spark…")
        spark.stop()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for Weather History Bronze writer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Historical batch (2023-01-01..2024-12-31) → single parquet
  python weather_bronze_cli.py

  # Historical with auto-download if missing
  python weather_bronze_cli.py --download-data

  # OOT single day (partitioned)
  python weather_bronze_cli.py --snapshotdate 2025-01-15

  # OOT with auto-download of historical if missing
  python weather_bronze_cli.py --snapshotdate 2025-01-15 --download-data
"""
    )
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=False,
        default=None,
        help="OOT date YYYY-MM-DD. If omitted, writes historical 2023-01-01..2024-12-31 as a single parquet."
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="If set, download missing raw NOAA ISD files into ../data/weather_history before processing."
    )
    parser.add_argument("--oot-start", type=str, default=None, help="Range start YYYY-MM-DD (optional).")
    parser.add_argument("--oot-end",   type=str, default=None, help="Range end YYYY-MM-DD (optional).")
    parser.add_argument("--no-hist",   action="store_true",    help="Skip historical write.")
    args = parser.parse_args()

    # Validate snapshotdate if provided
    if args.snapshotdate:
        try:
            datetime.strptime(args.snapshotdate, "%Y-%m-%d")
        except ValueError:
            print(f"ERROR: Invalid date format: {args.snapshotdate}")
            print("Expected: YYYY-MM-DD (e.g., 2025-01-15)")
            raise SystemExit(1)

    main(snapshotdate=args.snapshotdate, download_data=args.download_data)