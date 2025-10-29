# airports_silver_cli.py
import os, sys, time, argparse
from datetime import datetime
from pyspark.sql import SparkSession

# === 导入你已有的处理函数（改成你的真实导入路径）===
from utils.silver_table_airports import process_silver_airports_table  # TODO: 替换为正确路径

def _parse_csv_list(s: str | None):
    if not s:
        return None
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return vals or None

def _parse_bool(s: str | None):
    if s is None:
        return None
    s2 = s.strip().lower()
    if s2 in ("true", "1", "yes", "y"):
        return True
    if s2 in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid bool: {s} (expected true/false)")

def main(
    bronze_airport_directory: str,
    silver_airport_directory: str,
    freq_type_whitelist: list[str] | None,
    scheduled_only: bool | None,
    iata_targets: list[str] | None,
    wide_subdir: str,
    subset_subdir: str,
    driver_mem: str,
    executor_mem: str,
    shuffle_parts: int,
):
    t0 = time.time()
    print("\n" + "=" * 84)
    print("SILVER: US Airport Wide + Optional IATA Subset")
    print("=" * 84)
    print("Start Time         :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Bronze directory   :", bronze_airport_directory)
    print("Silver directory   :", silver_airport_directory)
    print("freq_type_whitelist:", freq_type_whitelist or "(no filter)")
    print("scheduled_only     :", scheduled_only if scheduled_only is not None else "(no filter)")
    print("iata_targets       :", iata_targets or "(no subset)")
    print("wide_subdir        :", wide_subdir)
    print("subset_subdir      :", subset_subdir)
    print("Spark              :", f"driver={driver_mem}, executor={executor_mem}, shuffle={shuffle_parts}")
    print("=" * 84 + "\n")

    # 基本校验
    if not os.path.isdir(bronze_airport_directory):
        print(f"ERROR: Bronze directory not found: {bronze_airport_directory}")
        sys.exit(1)
    os.makedirs(silver_airport_directory, exist_ok=True)

    # Spark
    spark = (
        SparkSession.builder
        .appName("AirportsSilver")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        ret = process_silver_airports_table(
            bronze_airport_directory=bronze_airport_directory,
            silver_airport_directory=silver_airport_directory,
            spark=spark,
            freq_type_whitelist=freq_type_whitelist,
            scheduled_only=scheduled_only,
            iata_targets=iata_targets,
            wide_subdir=wide_subdir,
            subset_subdir=subset_subdir
        )
        print("\n[SILVER] Output paths:")
        for k, v in ret.items():
            print(f"  - {k}: {v}")

        print(f"\n✓ Done in {(time.time() - t0):.1f}s")

        # 友好提示：快速预览行数
        try:
            if ret.get("wide_path") and os.path.exists(ret["wide_path"]):
                cnt = spark.read.parquet(ret["wide_path"]).count()
                print(f"[SILVER] US wide rows: {cnt}")
            if ret.get("subset_path") and ret["subset_path"] and os.path.exists(ret["subset_path"]):
                scnt = spark.read.parquet(ret["subset_path"]).count()
                print(f"[SILVER] subset rows: {scnt}")
        except Exception as e:
            print("[SILVER] Count preview failed (non-blocking):", e)

    finally:
        print("\nStopping Spark…")
        spark.stop()
        print("Stopped.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="CLI wrapper for process_silver_airports_table (no change to your process code).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # 1) 最常用：全美 US 宽表（不筛频率，不筛是否有定期航班），不生成子集
  python airports_silver_cli.py \
    --bronze-dir ../datamart/bronze/airport \
    --silver-dir ../datamart/silver/airport

  # 2) 只保留常见频率类型 + 仅有定期航班机场
  python airports_silver_cli.py \
    --bronze-dir ../datamart/bronze/airport \
    --silver-dir ../datamart/silver/airport \
    --freq-types TWR,APP,A/D,ATIS,AWOS,GND \
    --scheduled-only true

  # 3) 生成纽约三场子表（或你想要的任意 IATA 列表）
  python airports_silver_cli.py \
    --bronze-dir ../datamart/bronze/airport \
    --silver-dir ../datamart/silver/airport \
    --iata JFK,LGA,EWR

  # 4) 自定义子目录名（默认 US_airports / new_york）
  python airports_silver_cli.py \
    --bronze-dir ../datamart/bronze/airport \
    --silver-dir ../datamart/silver/airport \
    --wide-subdir US_airports \
    --subset-subdir new_york
"""
    )
    ap.add_argument("--bronze-dir", required=True, help="Path to Bronze airport root (contains 'airports', 'airport_frequencies' parquet).")
    ap.add_argument("--silver-dir", required=True, help="Path to Silver airport root.")
    ap.add_argument("--freq-types", default=None, help="Comma-separated whitelist, e.g. 'TWR,APP,A/D,ATIS,AWOS,GND'.")
    ap.add_argument("--scheduled-only", default=None, help="true/false to filter scheduled_service; omit for no filter.")
    ap.add_argument("--iata", default=None, help="Comma-separated IATA targets for subset, e.g. 'JFK,LGA,EWR'.")
    ap.add_argument("--wide-subdir", default="US_airports", help="US wide table subdir under silver dir.")
    ap.add_argument("--subset-subdir", default="new_york", help="Subset table subdir under silver dir.")
    ap.add_argument("--driver-mem", default="2g")
    ap.add_argument("--executor-mem", default="2g")
    ap.add_argument("--shuffle-parts", type=int, default=24)

    args = ap.parse_args()

    try:
        scheduled_only = _parse_bool(args.scheduled_only) if args.scheduled_only is not None else None
    except ValueError as e:
        print("ERROR:", e); sys.exit(1)

    main(
        bronze_airport_directory=args.bronze_dir,
        silver_airport_directory=args.silver_dir,
        freq_type_whitelist=_parse_csv_list(args.freq_types),
        scheduled_only=scheduled_only,
        iata_targets=_parse_csv_list(args.iata),
        wide_subdir=args.wide_subdir,
        subset_subdir=args.subset_subdir,
        driver_mem=args.driver_mem,
        executor_mem=args.executor_mem,
        shuffle_parts=args.shuffle_parts,
    )
