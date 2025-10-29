import os, glob, shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, List, Tuple
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType

def _write_historical_single(df_hist, output_root: str):
    """Write a single parquet file (coalesce + rename part-*)"""
    os.makedirs(output_root, exist_ok=True)
    tmp_hist = os.path.join(output_root, "_tmp_hist")

    cnt = df_hist.count()
    print(f"[historical] rows: {cnt}")
    if cnt == 0:
        shutil.rmtree(tmp_hist, ignore_errors=True)
        print("[historical] nothing to write")
        return

    df_hist.coalesce(1).write.mode("overwrite").parquet(tmp_hist)
    parts = glob.glob(os.path.join(tmp_hist, "part-*.parquet")) or glob.glob(os.path.join(tmp_hist, "*.parquet"))
    if not parts:
        shutil.rmtree(tmp_hist, ignore_errors=True)
        raise RuntimeError(f"No parquet part found in {tmp_hist}")

    final_path = os.path.join(output_root, "weather_history.parquet")
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(parts[0], final_path)
    shutil.rmtree(tmp_hist, ignore_errors=True)
    print(f"[historical] saved to: {final_path}")

def _write_oot_partitioned(df_oot,
    output_root: str,
    filename_prefix: str = "bronze_weather_history_",
    coalesce_threshold: int = 2_000_000,
    parallelism: int = 4,
) -> List[Tuple[str, int, str]]:
    """
    Write ONE Parquet file per day with a friendly name:
      {output_root}/{filename_prefix}YYYY_MM_DD.parquet

    Steps per day:
      filter → (optional) coalesce(1) → write to tmp dir → move 'part-*' to target filename → clean tmp.

    Returns a list of (date_str, row_count, final_path).
    """
    os.makedirs(output_root, exist_ok=True)
    tmp_root = os.path.join(output_root, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    # Cache to avoid re-reading for each day
    df_oot = df_oot.persist()
    _ = df_oot.count()

    # Precompute counts per day in ONE pass
    counts = (
        df_oot.groupBy("snapshot_date")
            .agg(F.count(F.lit(1)).alias("cnt"))
            .collect()
        )
    day_counts = { r["snapshot_date"]: int(r["cnt"]) for r in counts }
    dates_to_write = sorted(day_counts.keys())
    if not dates_to_write:
        print("[OOT] nothing to write")
        df_oot.unpersist()
        return []

    def _write_one(day) -> Tuple[str, int, str]:
        day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)
        cnt = day_counts.get(day, 0)
        if cnt == 0:
            return (day_str, 0, "")

        # Filter this day
        df_day = df_oot.filter(F.col("snapshot_date") == F.lit(day).cast(DateType()))
        tmp_dir = os.path.join(tmp_root, f"d={day_str}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Single-file write for small days
        df_out = df_day.coalesce(1) if cnt < coalesce_threshold else df_day
        df_out.write.mode("overwrite").parquet(tmp_dir)

        # Find the part file Spark wrote
        parts = glob.glob(os.path.join(tmp_dir, "part-*.parquet")) or glob.glob(os.path.join(tmp_dir, "*.parquet"))
        if not parts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"No parquet part found in {tmp_dir}")

        final_name = f"{filename_prefix}{day_str.replace('-', '_')}.parquet"
        final_path = os.path.join(output_root, final_name)
        if os.path.exists(final_path):
            os.remove(final_path)
        shutil.move(parts[0], final_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"saved: {final_path} ({cnt} rows)")
        return (day_str, cnt, final_path)

    # Write several days in parallel (each schedules a Spark job)
    results: List[Tuple[str, int, str]] = []
    max_workers = max(1, min(parallelism, len(dates_to_write)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_write_one, dates_to_write):
            results.append(res)

    df_oot.unpersist()
    return results

def process_bronze_weather_history(
    spark,
    raw_root_dir: str = "../../data/weather_history",
    output_root: str = "../../datamart/bronze/weather_history",
    oot_date_list: Optional[Iterable[str]] = None,   # e.g. ["2025-01-01", ...]; None => historical 2023–2024
    oot_start: Optional[str] = None,                 # optional range start
    oot_end: Optional[str] = None,                   # optional range end
    write_historical: bool = True,                   # default keeps current behavior
):
    """
    Historical mode (oot_date_list=None):
        Writes ONE parquet file: {output_root}/weather_history_2023_2024.parquet

    OOT mode (oot_date_list provided):
        Writes partitioned parquet by snapshot_date under: {output_root}/oot/
    """
    # 1) Read everything recursively
    df_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("recursiveFileLookup", True)
        .csv(raw_root_dir)
    )

    # 2) Parse NOAA DATE -> obs_ts -> snapshot_date; derive station_id from path
    base = os.path.basename(os.path.normpath(raw_root_dir))  # "weather_history"
    df = (
        df_raw
        .withColumn("obs_ts",
            F.to_timestamp(F.col("DATE").cast(StringType()), "yyyy-MM-dd'T'HH:mm:ss")
        )
        .withColumn("snapshot_date", F.to_date(F.col("obs_ts")))
        .withColumn("input_path", F.input_file_name())
        .withColumn("station_id", F.regexp_extract(F.col("input_path"), rf"/{base}/([^/]+)/", 1))
        .drop("input_path")
        .filter(F.col("snapshot_date").isNotNull())
    )

    if oot_date_list or (oot_start and oot_end):
        if oot_start and oot_end:
            start_d = F.lit(oot_start).cast(DateType())
            end_d   = F.lit(oot_end).cast(DateType())
            df_oot = df.filter((F.col("snapshot_date") >= start_d) & (F.col("snapshot_date") <= end_d))
        else:
            df_oot = df.filter(F.col("snapshot_date").isin([F.to_date(F.lit(d)) for d in oot_date_list]))
        _write_oot_partitioned(df_oot, output_root)
        return
    
    if write_historical:
        # Historical default: 2023-01-01 .. 2024-12-31
        hist_start = F.lit("2023-01-01").cast(DateType())
        hist_end   = F.lit("2024-12-31").cast(DateType())
        df_hist = df.filter((F.col("snapshot_date") >= hist_start) & (F.col("snapshot_date") <= hist_end))
        _write_historical_single(df_hist, output_root)

