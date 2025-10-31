import os, glob, shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, List, Tuple
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType
import os, glob, shutil
from datetime import date

def process_main_weather_spark(spark, data_path, bronze_dir, snapshotdate=None):
    
    files_list = [data_path+os.path.basename(f) for f in glob.glob(os.path.join(data_path, '*'))]
    df = spark.read.option("recursiveFileLookup", "true").option("header", "true").option("inferSchema", "true").csv(data_path)
    
    df = df.withColumn("extracted_date",F.to_timestamp(F.col("DATE").cast(StringType()), "yyyy-MM-dd'T'HH:mm:ss"))\
        .withColumn("extracted_date", F.to_date(F.col("extracted_date")))
    
    # helper to write one named parquet file
    def _write_single_parquet(spark_session, df_single, out_file_path):
        tmp_dir = out_file_path + ".__tmp"
        # start clean
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

        # 1) Spark writes to a temp directory (single part via coalesce)
        df_single.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

        # 2) Find the actual parquet part Spark produced
        parts = glob.glob(os.path.join(tmp_dir, "part-*.parquet")) \
            or glob.glob(os.path.join(tmp_dir, "*.parquet"))
        if not parts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"No parquet part found in {tmp_dir}")

        # 3) Move to final path (overwrite if exists)
        if os.path.exists(out_file_path):
            os.remove(out_file_path)
        shutil.move(parts[0], out_file_path)

        # 4) Remove temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5) Optional: sweep stray dotfiles next to outputs (e.g. ._*, .DS_Store)
        out_dir = os.path.dirname(out_file_path)
        for name in os.listdir(out_dir):
            if name.startswith(".") and name != ".gitignore":
                path = os.path.join(out_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    if snapshotdate:
        date_str = snapshotdate.strftime("%Y-%m-%d")
        df = df.filter(F.col("extracted_date") == date_str)
        df_d = df.drop("extracted_date")
        out_path = f"{bronze_dir}bronze_weather_store_{date_str}.parquet"
        _write_single_parquet(spark, df_d, out_path)
        print(f"saved {out_path}  rows={df_d.count()}")
    else:
        data_end_date = "2025-04-01"
        df = df.filter(F.col("extracted_date") <= F.lit(data_end_date))
        cutoff = date(2025, 1, 1)
        # One file for everything before cutoff
        df_pre = df.filter(F.col("extracted_date") < F.lit(cutoff)).drop("extracted_date")
        pre_count = df_pre.count()
        if pre_count > 0:
            pre_out = f"{bronze_dir}bronze_weather_store_2023_2024.parquet"
            _write_single_parquet(spark, df_pre, pre_out)
            print(f"saved: {pre_out}  rows={pre_count}")
        
        # One file per date (chronological) on/after cutoff
        df_post = df.filter(F.col("extracted_date") >= F.lit(cutoff))
        dates = (df_post.select("extracted_date")
                        .where(F.col("extracted_date").isNotNull())
                        .distinct()
                        .orderBy("extracted_date")      # ensures chronological generation
                        .collect())
                        
        for r in dates:
            d = r["extracted_date"]
            date_str = d.strftime("%Y-%m-%d")
            df_d = df_post.filter(F.col("date") == F.lit(d)).drop("extracted_date")
            out_path = f"{bronze_dir}bronze_weather_store_{date_str}.parquet"
            _write_single_parquet(spark, df_d, out_path)
            print(f"saved {out_path}  rows={df_d.count()}")



    