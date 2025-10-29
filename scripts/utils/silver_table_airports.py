# ===== Silver: US Airport Wide (canonicalized + arrays + has_* only)
#        + NYC subset (JFK/LGA/EWR) =====
import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, trim, upper, lower, regexp_replace, when, coalesce, concat_ws

def process_silver_airports_table(
    bronze_airport_directory: str,        # e.g. "datamart/bronze/airport"
    silver_airport_directory: str,        # e.g. "datamart/silver/airport"
    spark,
    freq_type_whitelist=None,             # e.g. ["TWR","APP","A/D","ATIS","AWOS","GND"]; None = no filter
    scheduled_only=None,                  # True/False to filter scheduled_service; None = no filter
    iata_targets=None,                    # e.g. ["JFK","LGA","EWR"]; None = don't make subset
    wide_subdir="US_airports",            # <- 全美 US 宽表子目录
    subset_subdir="new_york"              # <- 纽约三场子表子目录
):
    # 1) Load Bronze（并安全重命名避免冲突）
    airports = (
        spark.read.parquet(os.path.join(bronze_airport_directory, "airports"))
        .withColumnRenamed("id", "airport_id")
        .withColumnRenamed("type", "airport_type")
        .withColumn("ident_norm", upper(trim(col("ident"))))
        .withColumn("iso_country_norm", upper(trim(col("iso_country"))))
        .withColumn("scheduled_service_norm", upper(trim(col("scheduled_service"))))
        .dropDuplicates(["airport_id"])
    )
    freqs = (
        spark.read.parquet(os.path.join(bronze_airport_directory, "airport_frequencies"))
        .withColumnRenamed("id", "freq_id")
        .withColumnRenamed("type", "freq_type")
        .withColumn("airport_ident_norm", upper(trim(col("airport_ident"))))
        .withColumn("freq_type_norm", upper(trim(col("freq_type"))))
        .dropDuplicates(["freq_id"])
    )

    # 2) US 筛选 + 可选条件
    airports = airports.filter(col("iso_country_norm") == lit("US"))
    airports = airports.withColumn(
        "scheduled_service_bool",
        when(col("scheduled_service_norm") == "YES", lit(True))
        .when(col("scheduled_service_norm") == "NO",  lit(False))
        .otherwise(lit(None))
    )
    if scheduled_only is not None:
        airports = airports.filter(col("scheduled_service_bool") == lit(bool(scheduled_only)))

    # 频率只保留属于这些 US 机场的（半连接）
    freqs = freqs.join(
        airports.select(col("airport_id").alias("aid")),
        freqs.airport_ref == col("aid"),
        "left_semi"
    )
    if freq_type_whitelist:
        wl = [t.upper() for t in freq_type_whitelist]
        freqs = freqs.filter(col("freq_type_norm").isin(wl))

    # 3) Join → 明细 core（作为 pivot 来源）
    core = (
        freqs.join(airports, freqs.airport_ref == airports.airport_id, "left")
             .withColumn("_ident_match", when(col("airport_ident_norm") == col("ident_norm"), lit(1)).otherwise(lit(0)))
             .withColumn("has_geo", when(col("latitude_deg").isNull() | col("longitude_deg").isNull(), lit(False)).otherwise(lit(True)))
    )

    # 4) 强力归一化：把各种写法映射到少数 canonical 类型
    text = concat_ws(" ", coalesce(col("freq_type"), lit("")), coalesce(col("description"), lit("")))
    t = lower(regexp_replace(text, r"[\s\-]+", "_"))
    canonical = (core
        .withColumn("canonical_type",
            when(t.rlike(r"\b(twr|tower)\b"), lit("tower"))
            .when(t.rlike(r"\b(gnd|ground)\b"), lit("ground"))
            .when(t.rlike(r"\b(app|appr|aprch|approach)\b"), lit("approach"))
            .when(t.rlike(r"\b(dep|dept|departure)\b"), lit("departure"))
            .when(t.rlike(r"(a[_/]*d|app[_/]*dep|approach[_/]*departure)"), lit("approach_departure"))
            .when(t.rlike(r"\b(d[_]*atis|d-atis|d_atis|atis)\b"), lit("atis"))
            .when(t.rlike(r"\b(awos|wx[_]*asos|wxas|asos)\b"), lit("awos"))
            .when(t.rlike(r"(ctaf.*|.*ctaf|unicom)"), lit("ctaf_unicom"))
            .when(t.rlike(r"(cld|clr|clnc[_]*del|clrd|clrn|clearance)"), lit("clearance_delivery"))
            .when(t.rlike(r"\b(fss|rfss|rco)\b"), lit("fss"))
            .when(t.rlike(r"\b(cntr|ctr|center)\b"), lit("center"))
            .otherwise(lit("other"))
        )
    )

    # 5) Pivot（每机场一行；每类列为 array<double>）
    id_cols = [
        "airport_id","ident","name","airport_type","municipality",
        "iso_country","iso_region","continent",
        "latitude_deg","longitude_deg","elevation_ft","scheduled_service_bool",
        "_ident_match","has_geo"
    ]
    canonical_order = [
        "tower","ground","approach","departure","approach_departure",
        "atis","awos","ctaf_unicom","clearance_delivery","fss","center","other"
    ]
    wide = (canonical
        .select(*id_cols, "canonical_type", "frequency_mhz")
        .groupBy(*id_cols)
        .pivot("canonical_type", canonical_order)
        .agg(F.collect_set("frequency_mhz"))   # array<double>
    )
    # 列名加前缀 freq_
    for c in wide.columns:
        if c not in id_cols:
            wide = wide.withColumnRenamed(c, f"freq_{c}")

    # 只追加 has_*（不再生成 cnt_*）
    freq_cols = [c for c in wide.columns if c.startswith("freq_")]
    for c in freq_cols:
        wide = wide.withColumn(f"has_{c[5:]}", F.size(F.col(c)) > 0)

    # 6) 写出全美 US 宽表 -> datamart/silver/airport/US_airports
    us_out_path = os.path.join(silver_airport_directory, wide_subdir)
    os.makedirs(os.path.dirname(us_out_path), exist_ok=True)
    wide.write.mode("overwrite").parquet(us_out_path)
    print("[SILVER] US airports wide ->", us_out_path)

    subset_path = None
    # 7) 纽约三场（或任意 IATA 列表）子表 -> datamart/silver/airport/new_york
    if iata_targets:
        airports_dim = (spark.read.parquet(os.path.join(bronze_airport_directory, "airports"))
                        .select(col("id").alias("airport_id"),
                                upper(trim(col("iata_code"))).alias("iata_code"),
                                upper(trim(col("ident"))).alias("ident_bz")))
        wide_enriched = (wide.join(airports_dim, on="airport_id", how="left")
                             .withColumn("ident_up", upper(trim(col("ident")))))

        targets = sorted({t.upper().strip() for t in iata_targets})
        ident_targets = [f"K{t}" for t in targets]  # 美国机场 ICAO 兜底

        subset = wide_enriched.filter(
            (col("iata_code").isin(targets)) | (col("ident_up").isin(ident_targets))
        )

        subset_path = os.path.join(silver_airport_directory, subset_subdir)
        os.makedirs(os.path.dirname(subset_path), exist_ok=True)
        subset.write.mode("overwrite").parquet(subset_path)

        print(f"[SILVER] subset ({','.join(targets)}) ->", subset_path)
        subset.select("airport_id","ident","iata_code","name","iso_region").show(truncate=False)

    return {"wide_path": us_out_path, "subset_path": subset_path}


# ===== 示例调用（写到你指定的两个路径）=====
# bronze_airport = "datamart/bronze/airport"
# silver_airport = "datamart/silver/airport"
# ret = process_silver_airports_table(
#     bronze_airport_directory=bronze_airport,
#     silver_airport_directory=silver_airport,
#     spark=spark,
#     freq_type_whitelist=None,
#     scheduled_only=None,
#     iata_targets=["JFK","LGA","EWR"],         # 纽约三场
#     wide_subdir="US_airports",                # 全美 US 宽表子目录
#     subset_subdir="new_york"                  # 纽约三场子表子目录
# )
# print(ret)
# # 读回检查
# df_us = spark.read.parquet(ret["wide_path"])
# df_nyc = spark.read.parquet(ret["subset_path"])
# print("US rows:", df_us.count(), "NYC rows:", df_nyc.count())
