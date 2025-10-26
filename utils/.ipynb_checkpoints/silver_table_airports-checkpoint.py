# ===== Silver: US Airport Wide Table (canonicalized + arrays + has_* only) =====
import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, trim, upper, lower, regexp_replace, when, coalesce, concat_ws

def process_silver_airports_table(
    bronze_airport_directory: str,      # e.g. "datamart/bronze/airport"
    silver_airport_directory: str,      # e.g. "datamart/silver/airport"（直接写到此目录）
    spark,
    freq_type_whitelist=None,           # 例: ["TWR","APP","A/D","ATIS","AWOS","GND"]；None=不过滤
    scheduled_only=None                 # True/False 仅保留有/无定期航班；None=不过滤
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

    # 频率只保留属于这些 US 机场的（用数值主键半连接）
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

    # 6) 只追加 has_*（不再生成 cnt_*）
    freq_cols = [c for c in wide.columns if c.startswith("freq_")]
    for c in freq_cols:
        wide = wide.withColumn(f"has_{c[5:]}", F.size(F.col(c)) > 0)

    # 7) 写出（覆盖 silver 目录本身）
    os.makedirs(silver_airport_directory, exist_ok=True)
    out_path = silver_airport_directory
    wide.write.mode("overwrite").parquet(out_path)

    # 8) QC
    print("[SILVER] airport_wide_us (canonicalized, has_* only) ->", out_path)
    print("[QC] rows:", wide.count(), "cols:", len(wide.columns))
    wide.select("ident","name", "has_tower","has_approach","has_departure","has_atis","has_awos","has_ground","has_ctaf_unicom").show(10, truncate=False)

    return out_path







# ===== 示例调用 =====
# bronze_airport = "datamart/bronze/airport"
# silver_airport = "datamart/silver/airport"
# os.makedirs(silver_airport, exist_ok=True)
# path = process_silver_airports_table(
#     bronze_airport_directory=bronze_airport,
#     silver_airport_directory=silver_airport,
#     spark=spark,
#     freq_type_whitelist=["TWR","APP","ATIS","GND"],  # 全类型传 None
#     scheduled_only=None                               # 只保留有定期航班传 True
# )
# df = spark.read.parquet(path); print("rows:", df.count()); df.printSchema()
