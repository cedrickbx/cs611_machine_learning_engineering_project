import os
from pyspark.sql import functions

def process_bronze_airports_table(csv_file_path, bronze_directory, spark, table_name=None,
                                  mode="overwrite", partition_cols=None):
    """
    将静态参考数据 CSV 原样落到 Bronze（Parquet）。
    - 不做业务字段变换
    - 仅添加元数据列：_ingest_ts, _source_file, _ingest_date
    - 默认 overwrite：参考表通常全量覆盖
    """
    # 读源数据（尽量少假设，保留原样）
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv(csv_file_path))

    # 元数据列（可追溯）
    df = (df
          .withColumn("_ingest_ts", functions.current_timestamp())
          .withColumn("_source_file", functions.input_file_name())
          .withColumn("_ingest_date", functions.current_date()))

    # 计数
    print(f"[BRONZE] {csv_file_path} rows:", df.count())

    # 目标路径（按表名一个目录）
    if table_name is None:
        base = os.path.basename(csv_file_path)
        table_name = os.path.splitext(base)[0]
    out_dir = os.path.join(bronze_directory, table_name)

    # 写 Parquet（推荐）
    writer = df.write.mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(out_dir)

    print("saved to:", out_dir)
    return df
