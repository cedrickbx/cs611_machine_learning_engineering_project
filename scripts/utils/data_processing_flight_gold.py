"""
Gold Layer Processing for Flight Delay Data - Feature Engineering

Contains functions for feature engineering on Silver layer data:
1. Drop unnecessary columns
2. Create IS_DELAYED ordinal target variable
3. Create 3-hour time buckets
4. Create flight volume and extra flight features

These functions are imported by gold_flight_store.py for execution
"""

import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, when, concat_ws, count, avg, stddev, 
    substring, regexp_replace, udf, desc
)
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.window import Window


def drop_unnecessary_columns(df):
    """
    Drop columns not needed for Gold layer
    
    Removes:
    - CANCELLED, DIVERTED, CANCELLATION_CODE (already filtered out)
    - source_file, processing_timestamp, sort_time (metadata)
    - is_delayed_15 (will be replaced with IS_DELAYED)
    
    Args:
        df: Input Silver DataFrame
        
    Returns:
        DataFrame with columns dropped
    """
    print("\n  1. Dropping unnecessary columns...")
    
    initial_cols = len(df.columns)
    
    drop_cols = [
        'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
        'source_file', 'is_delayed_15', 'processing_timestamp', 'sort_time'
    ]
    
    # Only drop columns that exist
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    if existing_drop_cols:
        df = df.drop(*existing_drop_cols)
        print(f"    Dropped {len(existing_drop_cols)} columns: {existing_drop_cols}")
    else:
        print(f"    No columns to drop (already removed)")
    
    final_cols = len(df.columns)
    print(f"    Columns: {initial_cols} → {final_cols}")
    
    return df


def create_delay_categories(df):
    """
    Create IS_DELAYED ordinal target variable with 3 categories
    
    Categories based on both DEP_DELAY_NEW and ARR_DELAY_NEW:
    - Category 0: delay < 60 minutes
    - Category 1: delay 60-119 minutes
    - Category 2: delay >= 120 minutes
    
    Uses the MAXIMUM of departure and arrival delays
    
    Args:
        df: Input DataFrame with DEP_DELAY_NEW and ARR_DELAY_NEW
        
    Returns:
        DataFrame with IS_DELAYED column added (moved to extreme right)
    """
    print("\n  2. Creating IS_DELAYED ordinal categories...")
    
    from pyspark.sql.functions import greatest, coalesce
    
    # Get maximum delay between departure and arrival
    # Use coalesce to handle nulls (treat as 0)
    df = df.withColumn(
        "max_delay",
        greatest(
            coalesce(col("DEP_DELAY_NEW"), lit(0.0)),
            coalesce(col("ARR_DELAY_NEW"), lit(0.0))
        )
    )
    
    # Create ordinal categories
    df = df.withColumn(
        "IS_DELAYED",
        when(col("max_delay") < 60.0, 0)
        .when((col("max_delay") >= 60.0) & (col("max_delay") < 120.0), 1)
        .when(col("max_delay") >= 120.0, 2)
        .otherwise(0)
        .cast(IntegerType())
    )
    
    # Drop temporary column
    df = df.drop("max_delay")
    
    # Move IS_DELAYED to the end (reorder columns)
    other_cols = [c for c in df.columns if c != "IS_DELAYED"]
    df = df.select(other_cols + ["IS_DELAYED"])
    
    # Calculate and print distribution
    total_rows = df.count()
    delay_dist = df.groupBy("IS_DELAYED").count().orderBy("IS_DELAYED").collect()
    
    print("\n    IS_DELAYED Distribution:")
    print("    " + "-"*60)
    category_names = {
        0: "Category 0: delay < 60 minutes",
        1: "Category 1: delay 60-119 minutes", 
        2: "Category 2: delay >= 120 minutes"
    }
    
    for row in delay_dist:
        category = row["IS_DELAYED"]
        count = row["count"]
        pct = (count / total_rows) * 100
        print(f"    {category_names.get(category, f'Category {category}'):<40} {count:>8,} ({pct:>5.2f}%)")
    
    print("    " + "-"*60)
    print(f"    {'Total':<40} {total_rows:>8,} (100.00%)")
    
    return df


def create_3hour_buckets(df):
    """
    Create 3-hour time bucket columns for departure and arrival times
    
    Maps time blocks to 3-hour buckets:
    - 0001-0559 → 0000-0259, 0300-0559
    - 0600-0859 → 0600-0859
    - 0900-1159 → 0900-1159
    - etc.
    
    Args:
        df: Input DataFrame with DEP_TIME_BLK and ARR_TIME_BLK
        
    Returns:
        DataFrame with dep_3hour_col and arr_3hour_col added
    """
    print("\n  3. Creating 3-hour time buckets...")
    
    # Mapping from time blocks to 3-hour buckets
    # Note: Some blocks span multiple 3-hour periods, we map to the starting period
    time_block_to_3hour = {
        "0001-0559": "0000-0259",  # Early morning - map to first 3-hour bucket
        "0600-0659": "0600-0859",
        "0700-0759": "0600-0859",
        "0800-0859": "0600-0859",
        "0900-0959": "0900-1159",
        "1000-1059": "0900-1159",
        "1100-1159": "0900-1159",
        "1200-1259": "1200-1459",
        "1300-1359": "1200-1459",
        "1400-1459": "1200-1459",
        "1500-1559": "1500-1759",
        "1600-1659": "1500-1759",
        "1700-1759": "1500-1759",
        "1800-1859": "1800-2059",
        "1900-1959": "1800-2059",
        "2000-2059": "1800-2059",
        "2100-2159": "2100-2359",
        "2200-2259": "2100-2359",
        "2300-2359": "2100-2359"
    }
    
    # Create UDF for mapping
    from pyspark.sql.functions import udf
    
    def map_to_3hour(time_blk):
        if time_blk is None:
            return None
        # Handle the special case of 0001-0559
        if time_blk == "0001-0559":
            return "0000-0259"
        # Extract hour from start of time block
        if len(time_blk) >= 4:
            hour = int(time_blk[:2])
            # Map to 3-hour bucket
            if hour < 3:
                return "0000-0259"
            elif hour < 6:
                return "0300-0559"
            elif hour < 9:
                return "0600-0859"
            elif hour < 12:
                return "0900-1159"
            elif hour < 15:
                return "1200-1459"
            elif hour < 18:
                return "1500-1759"
            elif hour < 21:
                return "1800-2059"
            else:
                return "2100-2359"
        return None
    
    map_3hour_udf = udf(map_to_3hour, StringType())
    
    # Apply mapping
    df = df.withColumn("dep_3hour_col", map_3hour_udf(col("DEP_TIME_BLK")))
    df = df.withColumn("arr_3hour_col", map_3hour_udf(col("ARR_TIME_BLK")))
    
    # Validate mapping
    dep_buckets = df.select("dep_3hour_col").distinct().count()
    arr_buckets = df.select("arr_3hour_col").distinct().count()
    
    print(f"    Created dep_3hour_col: {dep_buckets} unique buckets")
    print(f"    Created arr_3hour_col: {arr_buckets} unique buckets")
    
    # Show sample mapping
    print("\n    Sample time block → 3-hour bucket mapping:")
    sample_mapping = df.select("DEP_TIME_BLK", "dep_3hour_col").distinct().orderBy("DEP_TIME_BLK").limit(5).collect()
    for row in sample_mapping:
        print(f"      {row['DEP_TIME_BLK']} → {row['dep_3hour_col']}")
    
    return df


def create_flight_volume_features(df, spark):
    """
    Create 7 features related to flight scheduling and volume patterns
    
    Features:
    1. flight_id: Concatenate carrier + flight number
    2. daily_flights: Count flights per carrier-origin-date
    3. volume_zscore: Z-score of daily_flights
    4. is_rare: Flag if flight_id appears < 3 times
    5. is_abnormal_num: Flag if flight number > 8000
    6. is_peak_day: Flag if volume_zscore > 2
    7. is_extra_candidate: Composite flag for extra flights
    
    Args:
        df: Input DataFrame
        spark: SparkSession
        
    Returns:
        DataFrame with 7 new features added
    """
    print("\n  4. Creating flight volume features...")
    
    # Feature 1: flight_id
    df = df.withColumn(
        "flight_id",
        concat_ws("_", col("OP_UNIQUE_CARRIER"), col("OP_CARRIER_FL_NUM"))
    )
    print("    ✓ Created flight_id")
    
    # Feature 2: daily_flights - count per carrier-origin-date
    window_daily = Window.partitionBy("OP_UNIQUE_CARRIER", "ORIGIN", "FlightDate")
    df = df.withColumn("daily_flights", count("*").over(window_daily))
    print("    ✓ Created daily_flights")
    
    # Feature 3: volume_zscore - z-score of daily_flights per carrier-origin
    # First compute mean and stddev for each carrier-origin pair
    carrier_origin_stats = df.groupBy("OP_UNIQUE_CARRIER", "ORIGIN").agg(
        avg("daily_flights").alias("mean_flights"),
        stddev("daily_flights").alias("stddev_flights")
    )
    
    # Join back to main dataframe
    df = df.join(carrier_origin_stats, on=["OP_UNIQUE_CARRIER", "ORIGIN"], how="left")
    
    # Calculate z-score
    df = df.withColumn(
        "volume_zscore",
        when(
            (col("stddev_flights").isNotNull()) & (col("stddev_flights") > 0),
            (col("daily_flights") - col("mean_flights")) / col("stddev_flights")
        ).otherwise(0.0)
    )
    
    # Drop temporary columns
    df = df.drop("mean_flights", "stddev_flights")
    print("    ✓ Created volume_zscore")
    
    # Feature 4: is_rare - flights appearing < 3 times
    flight_counts = df.groupBy("flight_id").agg(
        count("*").alias("flight_count")
    )
    df = df.join(flight_counts, on="flight_id", how="left")
    df = df.withColumn(
        "is_rare",
        when(col("flight_count") < 3, 1).otherwise(0).cast(IntegerType())
    )
    df = df.drop("flight_count")
    print("    ✓ Created is_rare")
    
    # Feature 5: is_abnormal_num - flight numbers > 8000
    df = df.withColumn(
        "is_abnormal_num",
        when(col("OP_CARRIER_FL_NUM") > 8000, 1).otherwise(0).cast(IntegerType())
    )
    print("    ✓ Created is_abnormal_num")
    
    # Feature 6: is_peak_day - volume_zscore > 2
    df = df.withColumn(
        "is_peak_day",
        when(col("volume_zscore") > 2.0, 1).otherwise(0).cast(IntegerType())
    )
    print("    ✓ Created is_peak_day")
    
    # Feature 7: is_extra_candidate - composite flag
    # Criteria: is_rare OR is_abnormal_num OR (is_rare AND is_peak_day) OR (is_abnormal_num AND IsPublicHoliday)
    df = df.withColumn(
        "is_extra_candidate",
        when(
            (col("is_rare") == 1) |
            (col("is_abnormal_num") == 1) |
            ((col("is_rare") == 1) & (col("is_peak_day") == 1)) |
            ((col("is_abnormal_num") == 1) & (col("IsPublicHoliday") == 1)),
            1
        ).otherwise(0).cast(IntegerType())
    )
    print("    ✓ Created is_extra_candidate")
    
    # Print summary statistics
    print("\n    Flight Volume Features Summary:")
    print("    " + "-"*60)
    
    total_rows = df.count()
    rare_count = df.filter(col("is_rare") == 1).count()
    abnormal_count = df.filter(col("is_abnormal_num") == 1).count()
    peak_count = df.filter(col("is_peak_day") == 1).count()
    extra_count = df.filter(col("is_extra_candidate") == 1).count()
    
    print(f"    is_rare (< 3 appearances):        {rare_count:>8,} ({rare_count/total_rows*100:>5.2f}%)")
    print(f"    is_abnormal_num (flight# > 8000): {abnormal_count:>8,} ({abnormal_count/total_rows*100:>5.2f}%)")
    print(f"    is_peak_day (z-score > 2):        {peak_count:>8,} ({peak_count/total_rows*100:>5.2f}%)")
    print(f"    is_extra_candidate:               {extra_count:>8,} ({extra_count/total_rows*100:>5.2f}%)")
    print("    " + "-"*60)
    
    return df

## Added gold validation 
def validate_gold_parquet(gold_output_path, spark, is_daily=False):
    """
    Validate Gold layer parquet file
    """
    print("\nValidating Gold layer output...")
    
    df_gold = spark.read.parquet(gold_output_path)
    
    validation = {
        'total_rows': df_gold.count(),
        'total_columns': len(df_gold.columns),
        'required_features': [
            'IS_DELAYED',
            'dep_3hour_col',
            'arr_3hour_col',
            'daily_flights',
            'volume_zscore',
            'is_rare',
            'is_abnormal_num',
            'is_peak_day',
            'is_extra_candidate'
        ]
    }
    
    # Verify required columns exist
    missing_cols = [col for col in validation['required_features'] 
                   if col not in df_gold.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required Gold columns: {missing_cols}")
    
    print("✓ Gold validation complete")
    return validation

