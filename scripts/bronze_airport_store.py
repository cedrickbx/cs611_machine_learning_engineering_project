# airports_bronze_cli.py
import os, sys, time, argparse
from datetime import datetime
from pyspark.sql import SparkSession

from utils.bronze_table_airports import process_bronze_airports_table  # 改成你的导入

def main(csv_path, bronze_dir, table_name, mode, partition_cols):
    print("\n" + "="*70)
    print("AIRPORTS CSV → BRONZE (Reference Table)")
    print("="*70)
    print("Start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("CSV  :", csv_path)
    print("Out  :", bronze_dir)
    print("Table:", table_name or "(from CSV filename)")
    print("Mode :", mode)
    print("Part :", partition_cols or "(none)")
    print("="*70 + "\n")

    if not os.path.isfile(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    os.makedirs(bronze_dir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("AirportsBronze")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    try:
        process_bronze_airports_table(
            csv_file_path=csv_path,
            bronze_directory=bronze_dir,
            spark=spark,
            table_name=table_name,
            mode=mode,                 # 默认 overwrite
            partition_cols=partition_cols
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bronze-dir", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    ap.add_argument("--partition-cols", default=None,
                    help="Comma-separated columns; usually empty for reference tables.")
    args = ap.parse_args()

    parts = [c.strip() for c in args.partition_cols.split(",")] if args.partition_cols else None
    main(args.csv, args.bronze_dir, args.table, args.mode, parts)
