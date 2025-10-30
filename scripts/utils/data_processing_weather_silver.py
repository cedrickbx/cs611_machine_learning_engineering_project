from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from datetime import date
from pyspark.sql import Window 
import os, glob, shutil

def process_main_weather_spark(spark, bronze_dir, silver_dir):
    files_list = [os.path.join(bronze_dir, os.path.basename(f))
                  for f in glob.glob(os.path.join(bronze_dir, '*'))]
    if not files_list:
        raise FileNotFoundError(f"No files found under {bronze_dir}")
    df = spark.read.parquet(*files_list)

    # df = df.filter(F.col("QUALITY_CONTROL") == F.lit("V020")) #filter for those records that have undergone quality control
    
    columns_to_keep = ['DATE', 'NAME', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP']
    df = df.select(*columns_to_keep)

    df = df.withColumn("time", F.date_format(F.to_timestamp("DATE"), "HH:mm"))\
        .withColumn("timestamp", F.to_timestamp("DATE"))\
        .withColumn("date", F.to_date(F.col("DATE")))

    #remove duplicates
    w_dedupe = Window.partitionBy("NAME", "date", "time").orderBy(F.col("timestamp").desc())
    df = df.withColumn("row", F.row_number().over(w_dedupe)) \
        .filter(F.col("row") == 1) \
        .drop("row")\
        .drop("timestamp")

    def _mask_missing(colname, codes):
        if isinstance(codes, (list, tuple, set)):
            return F.when(F.col(colname).isin(*codes), F.lit(None)).otherwise(F.col(colname))
        else:
            return F.when(F.col(colname) == F.lit(codes), F.lit(None)).otherwise(F.col(colname))

    def _split(colname):
        return F.split(F.col(colname).cast("string"), ",")

    def _part(arr_col, i, cast_type=None):
        c = arr_col.getItem(i)
        return c.cast(cast_type) if cast_type else c

    # --- Wind
    if "WND" in df.columns:
        arr = _split("WND")
        ok = F.size(arr) >= F.lit(5)
        df = df.withColumn("wind_dir_deg_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("wind_speed_tenths_tmp", F.when(ok, _part(arr, 3, "int")))\
              .withColumn("wind_dir_deg", _mask_missing("wind_dir_deg_tmp", 999))\
              .withColumn("wind_type", F.when(ok, _part(arr, 2)))\
              .withColumn("wind_speed_mps", _mask_missing("wind_speed_tenths_tmp", 9999) / F.lit(10.0))\
              .withColumn("wind_dir_qc", F.when(ok, _part(arr, 1)))\
              .withColumn("wind_speed_qc", F.when(ok, _part(arr, 4)))\
              .drop("wind_dir_deg_tmp", "wind_speed_tenths_tmp")

    # --- ceiling
    if "CIG" in df.columns:
        arr = _split("CIG")
        ok = F.size(arr) >= F.lit(4)
        df = df.withColumn("ceiling_m_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("ceiling_m", _mask_missing("ceiling_m_tmp", 99999))\
              .withColumn("ceiling_code", F.when(ok, _part(arr, 2)))\
              .withColumn("ceiling_qc", F.when(ok, _part(arr, 1)))\
              .drop("ceiling_m_tmp")

    # --- visibility
    if "VIS" in df.columns:
        arr = _split("VIS")
        ok = F.size(arr) >= F.lit(4)
        df = df.withColumn("visibility_m_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("visibility_m", _mask_missing("visibility_m_tmp", 999999))\
              .withColumn("visibility_var_code", F.when(ok, _part(arr, 2)))  \
              .withColumn("visibility_qc", F.when(ok, _part(arr, 1)))\
              .drop("visibility_m_tmp")

    # --- air temperature
    if "TMP" in df.columns:
        arr = _split("TMP")
        ok = F.size(arr) >= F.lit(2)
        df = df.withColumn("temp_tenths_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("temp_c", _mask_missing("temp_tenths_tmp", 9999) / F.lit(10.0))\
              .withColumn("temp_qc", F.when(ok, _part(arr, 1)))\
              .drop("temp_tenths_tmp")

    # --- Dew point
    if "DEW" in df.columns:
        arr = _split("DEW")
        ok = F.size(arr) >= F.lit(2)
        df = df.withColumn("dew_tenths_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("dewpoint_c", _mask_missing("dew_tenths_tmp", 9999) / F.lit(10.0))\
              .withColumn("dewpoint_qc", F.when(ok, _part(arr, 1)))\
              .drop("dew_tenths_tmp")

    # --- Sea Level Pressure
    if "SLP" in df.columns:
        arr = _split("SLP")
        ok = F.size(arr) >= F.lit(2)
        df = df.withColumn("slp_tenths_tmp", F.when(ok, _part(arr, 0, "int")))\
              .withColumn("slp_hpa", _mask_missing("slp_tenths_tmp", 99999) / F.lit(10.0))\
              .withColumn("slp_qc", F.when(ok, _part(arr, 1)))\
              .drop("slp_tenths_tmp")

    # Drop original groups
    drop_groups = [c for c in ["WND", "CIG", "VIS", "TMP", "DEW", "SLP"] if c in df.columns]
    df = df.drop(*drop_groups)

    # QC filters 
    def _as_str(c): 
        return F.col(c).cast("string")
    qc_1 = ["1", "5"]
    qc_2 = ["1", "5", "I", "C"]

    for qc in ['wind_dir_qc', 'wind_speed_qc', 'ceiling_qc', 'visibility_qc', 'slp_qc']:
        df = df.filter(_as_str(qc).isin(*qc_1))
    for qc in ['temp_qc', 'dewpoint_qc']:
        df = df.filter(_as_str(qc).isin(*qc_2))
    qc_drop = [c for c in ['temp_qc','dewpoint_qc','wind_dir_qc','wind_speed_qc','ceiling_qc','visibility_qc','slp_qc'] if c in df.columns]
    df = df.drop(*qc_drop)

    # ---- cast map
    cast_map = {
        'NAME': StringType(),
        'wind_dir_deg': FloatType(),
        'wind_type': StringType(),
        'wind_speed_mps': FloatType(),
        'ceiling_m': FloatType(),
        'ceiling_code': StringType(), #how ceiling is estimated
        'visibility_m': FloatType(),
        'visibility_var_code': StringType(), 
        'temp_c': FloatType(),
        'dewpoint_c': FloatType(),
        'slp_hpa': FloatType(),
        'date': DateType(),
        'time': StringType(),  # "HH:mm"
    }
    for colname, new_type in cast_map.items():
        df = df.withColumn(colname, F.col(colname).cast(new_type))

    desired_times = ["00:00","03:00","06:00","09:00","12:00","15:00","18:00","21:00"]

    #gatecheck - ensure all data has all 3-hr intervals
    # base grid of all (NAME, date, time)
    base = df.select("NAME", "date").distinct()
    grid = base.withColumn("time", F.explode(F.array([F.lit(t) for t in desired_times])))

    # left-join original data onto grid
    full = (grid.join(df, on=["NAME", "date", "time"], how="left"))

    # forward/backward fill within (NAME, date) ordered by hour
    full = full.withColumn("hour_int", F.substring("time", 1, 2).cast("int"))

    w_fwd  = Window.partitionBy("NAME", "date").orderBy("hour_int").rowsBetween(Window.unboundedPreceding, 0)
    w_back = Window.partitionBy("NAME", "date").orderBy(F.col("hour_int").desc()).rowsBetween(Window.unboundedPreceding, 0)

    cols_to_fill = [
        "wind_dir_deg","wind_type","wind_speed_mps",
        "ceiling_m","ceiling_code","visibility_m","visibility_var_code",
        "temp_c","dewpoint_c","slp_hpa"
    ]
    for c in cols_to_fill:
        ffill = F.last(F.col(c), ignorenulls=True).over(w_fwd)
        bfill = F.last(F.col(c), ignorenulls=True).over(w_back)
        full  = full.withColumn(c, F.coalesce(ffill, bfill))

    df = full.drop("hour_int")  # continue downstream with the augmented frame

    # encode airport names into values
    df = df.withColumn("NAME", 
                        F.when(F.col("NAME") == "JFK INTERNATIONAL AIRPORT, NY US", "JFK") 
                        .when(F.col("NAME") == "LAGUARDIA AIRPORT, NY US", "LGA") 
                        .when(F.col("NAME") == "NEWARK LIBERTY INTERNATIONAL AIRPORT, NJ US", "EWR"))

    # --- wind_type: 'N','C','9', None  (treat '9'/null as missing)
    df = (df.withColumn("wind_type",
            F.when(F.col("wind_type") == "A", 0) # Abridged beautfort (Simplified version of Beautfort)
            .when(F.col("wind_type") == "B", 1) # Beaufort (qualitative, empirical measure of wind speed based on observations of its effects on the sea and land - non-instrumental)
            .when(F.col("wind_type") == "C", 2) # Calm
            .when(F.col("wind_type") == "H", 3) # 5-minute Average Speed
            .when(F.col("wind_type") == "N", 4) # Normal
            .when(F.col("wind_type") == "R", 5) # 60-minute average speed
            .when(F.col("wind_type") == "Q", 6) #Squall
            .when(F.col("wind_type") == "T", 7) # 180-minute average speed
            .when(F.col("wind_type") == "V", 8) #Variable
            .when(F.col("wind_type").isNull(), 9) 
            .otherwise(9)    # missing/null values
            .cast("int")
        )
    )

    # --- ceiling_code: 'W','M','9', None  (domain depends on ISD; here just consistent ints)
    df = (df.withColumn("ceiling_code",
            F.when(F.col("ceiling_code") == "A", 0) #Aircraft
            .when(F.col("ceiling_code") == "B", 1) #Balloon
            .when(F.col("ceiling_code") == "C", 2) #Derived statistically
            .when(F.col("ceiling_code") == "E", 3) #Estimated
            .when(F.col("ceiling_code") == "M", 4) #Measured
            .when(F.col("ceiling_code") == "R", 5) #Radar
            .when(F.col("ceiling_code") == "S", 6) #ASOS augmented
            .when(F.col("ceiling_code") == "W", 7) #Obscured (Unknown)
            .when(F.col("ceiling_code").isNull(), 9)
            .otherwise(9)    # missing values
            .cast("int")
        )
    )

    # --- visibility_var_code: 'N','9', None
    df = (df
        .withColumn(
            "visibility_var_code_idx",
            F.when(F.col("visibility_var_code") == "N", 0) #Not variable
            .when(F.col("visibility_var_code") == "V", 1) #Variable
            .when(F.col("visibility_var_code").isNull(), 9)
            .otherwise(F.col("visibility_var_code"))    # missing values
            .cast("int")
        )
    )

    # frame data as forecast
    df = df.withColumn("T+1_forecast", F.col("date"))\
        .withColumn("date", F.date_add(F.col("date"), -1))\
        .withColumn("name", F.col("NAME"))  

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
    
    columns_in_front = ["name", "date", "time", "T+1_forecast"]
    other_columns = [col for col in df.columns if col not in columns_in_front]
    df = df.select(*columns_in_front, *other_columns)

    # ---------- WRITE ----------
    cutoff = date(2025, 1, 1)

    # One file for everything before cutoff
    df_pre = df.filter(F.col("date") < F.lit(cutoff))
    pre_count = df_pre.count()
    if pre_count > 0:
        pre_out = f"{silver_dir}silver_weather_store_2023_2024.parquet"
        _write_single_parquet(spark, df_pre, pre_out)
        print(f"saved: {pre_out}  rows={pre_count}")

    # One file per date (chronological) on/after cutoff
    df_post = df.filter(F.col("date") >= F.lit(cutoff))
    dates = (df_post.select("date")
                    .where(F.col("date").isNotNull())
                    .distinct()
                    .orderBy("date")      # ensures chronological generation
                    .collect())

    for r in dates:
        d = r["date"]
        date_str = d.strftime("%Y-%m-%d")
        df_d = df_post.filter(F.col("date") == F.lit(d))
        out_path = f"{silver_dir}silver_weather_store_{date_str}.parquet"
        _write_single_parquet(spark, df_d, out_path)
        print(f"saved (≥ 2025-01-01): {out_path}  rows={df_d.count()}")

    return df
