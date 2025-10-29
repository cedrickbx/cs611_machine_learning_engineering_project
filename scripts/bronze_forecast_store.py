import argparse, os
from datetime import datetime
from typing import List, Tuple, Iterable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from utils.data_processing_forecast_bronze import (
    fetch_gfs_points, _write_named_per_day, _write_single_file,
    DEFAULT_OUTPUT_ROOT, DEFAULT_FORECAST_HOURS,
    AIRPORTS
)
"""
CLI entrypoint that imports the core functions and produces the Bronze forecast tables.

Default:
- Historical: 2023-01-01 → 2024-12-31 → single Parquet
- OOT:        2025-01-01 → 2025-04-30 → named-per-day Parquets
"""

def process_bronze_gfs(
    spark: SparkSession,
    historical_start: str = "2023-01-01",
    historical_end:   str = "2024-12-31",   # keep historical strictly through Dec 2024
    oot_start:        str = "2025-01-01",
    oot_end:          str = "2025-03-31",
    output_root:      str = DEFAULT_OUTPUT_ROOT,
    fhours:           Iterable[int] = DEFAULT_FORECAST_HOURS,
    cycles:           Iterable[int] = (0, 6, 12, 18),
    airports=AIRPORTS,
    write_historical: bool = True
):
    """
    Build Bronze tables:
      - Historical: single Parquet file `forecast_2023_2024.parquet`
      - OOT: named-per-day files `forecast_YYYY_MM_DD.parquet`
    """
    os.makedirs(output_root, exist_ok=True)

    # ---- Historical (one file)
    if write_historical:
        hs, he = datetime.fromisoformat(historical_start), datetime.fromisoformat(historical_end)
        pdf_hist = fetch_gfs_points(hs, he, cycles=cycles, fhours=fhours,
                                    airports=airports)
        if pdf_hist.empty:
            print("[historical] nothing to write")
        else:
            df_hist = spark.createDataFrame(pdf_hist).withColumn("valid_date", F.to_date("valid_time"))
            hist_file = os.path.join(output_root, "forecast_2023_2024.parquet")
            _write_single_file(df_hist, hist_file)

    # ---- OOT (named per day)
    oot_dir = os.path.join(output_root, "oot_named")
    os.makedirs(oot_dir, exist_ok=True)
    oot_s, oot_e = datetime.fromisoformat(oot_start), datetime.fromisoformat(oot_end)
    pdf_oot = fetch_gfs_points(oot_s, oot_e, cycles=cycles, fhours=fhours,
                               airports=airports)
    if pdf_oot.empty:
        print("[OOT] nothing to write")
    else:
        df_oot = spark.createDataFrame(pdf_oot).withColumn("valid_date", F.to_date("valid_time"))
        _write_named_per_day(df_oot, oot_dir, filename_prefix="forecast")

    return True

def _parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser(description="Bronze writer: GFS 0.25° forecasts for JFK/LGA/EWR")
    p.add_argument("--hist-start", default="2023-01-01")
    p.add_argument("--hist-end",   default="2024-12-31")
    p.add_argument("--oot-start",  default=None)  # optional now
    p.add_argument("--oot-end",    default=None)  # optional now
    p.add_argument("--week-start", default=None,  help="Anchor date (YYYY-MM-DD) for 'next 7 days' weekly run")
    p.add_argument("--out",        default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    p.add_argument("--fhours",     default="6,12,24,48,72", help="CSV of forecast hours")
    p.add_argument("--cycles",     default="0,6,12,18",     help="CSV of cycles in UTC (00/06/12/18)")
    p.add_argument("--no-hist",    action="store_true", help="Skip historical write on this run") 
    args = p.parse_args()

    # Validate dates if provided
    for d in filter(None, [args.hist_start, args.hist_end, args.oot_start, args.oot_end, args.week_start]):
        datetime.strptime(d, "%Y-%m-%d")

    # Weekly mode?  --week-start takes precedence over --oot-*
    if args.week_start:
        if args.fhours == p.get_default("fhours"):
            args.fhours = "24,48,72,96,120,144,168"
        if args.cycles == p.get_default("cycles"):
            args.cycles = "0,12"
        oot_start = args.week_start
        oot_end   = args.week_start
        write_historical = not args.no_hist and False   # default skip in weekly mode
    else:
        oot_start = args.oot_start
        oot_end   = args.oot_end
        write_historical = not args.no_hist

    fhours = _parse_int_list(args.fhours)
    cycles = _parse_int_list(args.cycles)

    spark = (SparkSession.builder
             .appName("GFSBronzeAirports")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "24")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    try:
        process_bronze_gfs(
            spark=spark,
            historical_start=args.hist_start,
            historical_end=args.hist_end,
            oot_start=oot_start or "2025-01-01",  # harmless defaults if not used
            oot_end=oot_end or "2025-01-01",
            output_root=args.out,
            fhours=fhours,
            cycles=cycles,
            write_historical=write_historical, 
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    main()