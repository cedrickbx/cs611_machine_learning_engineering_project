import os, glob, shutil
from datetime import datetime, timedelta
from typing import Iterable, Optional, List

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType

# ---------- helpers ----------
def generate_dates(start_date_str: str, end_date_str: str) -> List[str]:
    """Generate inclusive daily date strings YYYY-MM-DD."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


# ---------- main ----------
def process_bronze_weather_history(
    spark,
    raw_root_dir: str = "data/weather_history",                  # your input tree (station_id/year/*.csv)
    output_root: str = "datamart/bronze/weather_history", # where we write outputs
    oot_date_list: Optional[Iterable[str]] = None,        # daily dates for OOT, e.g. generate_dates("2025-01-01","2025-03-31")
):
    # 0) Validate OOT date list if provided
    if oot_date_list:
        for d in oot_date_list:
            datetime.strptime(d, "%Y-%m-%d")

    # 1) Read all CSVs recursively and parse NOAA DATE -> snapshot_date
    df_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("recursiveFileLookup", True)
        .csv(raw_root_dir)
    )

    base = os.path.basename(os.path.normpath(raw_root_dir))
    df = (
        df_raw
        .withColumn("obs_ts", F.to_timestamp(F.col("DATE").cast(StringType()),
                                             "yyyy-MM-dd'T'HH:mm:ss"))
        .withColumn("snapshot_date", F.to_date(F.col("obs_ts")))
        .withColumn("input_path", F.input_file_name())
        .withColumn("station_id", F.regexp_extract(F.col("input_path"), rf"/{base}/([^/]+)/", 1))
        .drop("input_path")
        .filter(F.col("snapshot_date").isNotNull())
    )

    # 2) Split into historical (2023-01-01..2024-12-31) and OOT (2025-01-01..2025-03-31 or provided oot_date_list)
    hist_start = F.lit("2023-01-01").cast(DateType())
    hist_end   = F.lit("2024-12-31").cast(DateType())
    oot_start  = F.lit("2025-01-01").cast(DateType())
    oot_end    = F.lit("2025-03-31").cast(DateType())

    df_hist = df.filter((F.col("snapshot_date") >= hist_start) & (F.col("snapshot_date") <= hist_end))

    if oot_date_list:
        # exact daily list provided
        df_oot = df.filter(F.col("snapshot_date").isin([F.to_date(F.lit(d)) for d in oot_date_list]))
    else:
        # default to Jan–Mar 2025
        df_oot = df.filter((F.col("snapshot_date") >= oot_start) & (F.col("snapshot_date") <= oot_end))

    # ---------- Write historical as a single Parquet file ----------
    # Spark writes folders; to truly have a *single file*, write to a tmp dir with coalesce(1) and rename part-*.parquet
    hist_cnt = df_hist.count()
    print(f"[historical] rows: {hist_cnt}")
    os.makedirs(output_root, exist_ok=True)
    tmp_hist = os.path.join(output_root, "_tmp_hist")
    if hist_cnt > 0:
        # write one part file
        df_hist.coalesce(1).write.mode("overwrite").parquet(tmp_hist)
        # move/rename to friendly name
        parts = glob.glob(os.path.join(tmp_hist, "part-*.parquet")) or glob.glob(os.path.join(tmp_hist, "*.parquet"))
        if not parts:
            raise RuntimeError(f"No parquet part found in {tmp_hist}")
        hist_path = os.path.join(output_root, "weather_history_2023_2024.parquet")
        if os.path.exists(hist_path):
            os.remove(hist_path)
        shutil.move(parts[0], hist_path)
        shutil.rmtree(tmp_hist, ignore_errors=True)
        print(f"[historical] saved to: {hist_path}")
    else:
        shutil.rmtree(tmp_hist, ignore_errors=True)
        print("[historical] nothing to write")

    # ---------- Write OOT partitioned by day ----------
    oot_cnt = df_oot.count()
    print(f"[OOT] rows: {oot_cnt}")
    if oot_cnt > 0:
        oot_out_root = os.path.join(output_root, "oot")
        (
            df_oot
            .repartition("snapshot_date")  # good practice before partitioned write
            .write
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .partitionBy("snapshot_date")
            .parquet(oot_out_root)
        )
        # small log
        wrote_dates = [r["snapshot_date"].strftime("%Y-%m-%d") for r in df_oot.select("snapshot_date").distinct().collect()]
        sample = ", ".join(sorted(wrote_dates)[:10]) + (" …" if len(wrote_dates) > 10 else "")
        print(f"[OOT] saved under {oot_out_root}/snapshot_date=YYYY-MM-DD/ for dates: {sample}")
    else:
        print("[OOT] nothing to write")

    return df_hist, df_oot
