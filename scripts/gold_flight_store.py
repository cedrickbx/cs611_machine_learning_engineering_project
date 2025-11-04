"""
Main Script: Flight Gold Layer Processing
Process flight data from Silver to Flight Gold layer with feature engineering:
1. Historical batch processing (24 months)
2. Daily OOT processing (single day)

Usage:
    # Historical batch (default)
    python gold_flight_store.py
    
    # Daily OOT processing
    python gold_flight_store.py --snapshotdate 2025-01-15

Output:
    Historical: datamart/gold/flight/flight_gold_historical.parquet/
    Daily OOT:  datamart/gold/flight/flight_gold_oot_YYYY_MM_DD.parquet/
"""

import os
import time
import argparse
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sys

# Import our processing functions
from utils.data_processing_flight_gold import (
    drop_unnecessary_columns,
    create_delay_categories,
    create_3hour_buckets,
    create_flight_volume_features
)


def process_silver_to_flight_gold(silver_path, flight_gold_output_path, spark, 
                                   snapshot_date=None, output_mode='overwrite'):
    """
    Process Silver layer to Flight Gold layer with feature engineering
    
    Args:
        silver_path: Path to Silver Parquet
        flight_gold_output_path: Output path for Flight Gold Parquet
        spark: SparkSession
        snapshot_date: Optional date for OOT filtering
        output_mode: Write mode ('overwrite' or 'append')
        
    Returns:
        Processed Flight Gold DataFrame
    """
    print(f"\n{'='*80}")
    print("SILVER → FLIGHT GOLD PROCESSING")
    print("="*80)
    
    # Load Silver data
    print(f"\n  Loading Silver data from: {silver_path}")
    df = spark.read.parquet(silver_path)
    
    # Filter by snapshot_date if provided
    if snapshot_date:
        print(f"  Filtering for snapshot_date: {snapshot_date}")
        if 'snapshot_date' in df.columns:
            df = df.filter(col("snapshot_date") == snapshot_date)
        else:
            # If snapshot_date doesn't exist, add it
            from pyspark.sql.functions import lit, to_date
            df = df.withColumn("snapshot_date", to_date(lit(snapshot_date)))
            print(f"    Added snapshot_date column: {snapshot_date}")
    else:
        # Historical mode - ensure snapshot_date column exists
        if 'snapshot_date' not in df.columns:
            # If snapshot_date doesn't exist, derive from FlightDate
            if 'FlightDate' in df.columns:
                df = df.withColumn("snapshot_date", col("FlightDate"))
                print(f"    Created snapshot_date from FlightDate")
            else:
                print("    ⚠ WARNING: No snapshot_date or FlightDate column found")
    
    initial_count = df.count()
    initial_cols = len(df.columns)
    print(f"  Initial rows: {initial_count:,}")
    print(f"  Initial columns: {initial_cols}")
    
    # Apply feature engineering transformations
    print(f"\n  Applying feature engineering...")
    
    # Step 1: Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    
    # Step 2: Create delay categories
    df = create_delay_categories(df)
    
    # Step 3: Create 3-hour time buckets
    df = create_3hour_buckets(df)
    
    # Step 4: Create flight volume features
    df = create_flight_volume_features(df, spark)
    
    final_count = df.count()
    final_cols = len(df.columns)
    
    print(f"\n  Feature Engineering Complete:")
    print(f"    Final rows: {final_count:,}")
    print(f"    Final columns: {final_cols}")
    print(f"    Columns added: {final_cols - initial_cols + 7}")  # +7 because we dropped 7
    
    # Verify snapshot_date exists before saving
    if 'snapshot_date' not in df.columns:
        raise ValueError("snapshot_date column missing - cannot partition output!")
    
    # Save Flight Gold layer
    print(f"\n  Saving Flight Gold to: {flight_gold_output_path}")
    
    # Partition by snapshot_date
    df.write.mode(output_mode).partitionBy("snapshot_date").parquet(flight_gold_output_path)
    
    print(f"  ✓ Flight Gold saved successfully")
    
    return df


def validate_flight_gold_parquet(flight_gold_output_path, spark, is_daily=False):
    """
    Validate Flight Gold Parquet output
    
    Args:
        flight_gold_output_path: Path to Flight Gold Parquet
        spark: SparkSession
        is_daily: Whether this is daily OOT or historical batch
        
    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*80}")
    print("FLIGHT GOLD VALIDATION CHECKS")
    print("="*80)
    
    # Read Flight Gold Parquet
    df = spark.read.parquet(flight_gold_output_path)
    
    validation_results = {}
    
    # 1. Row count
    total_rows = df.count()
    validation_results['total_rows'] = total_rows
    print(f"\n1. Total Rows: {total_rows:,}")
    
    # 2. Column count
    total_cols = len(df.columns)
    validation_results['total_columns'] = total_cols
    print(f"\n2. Total Columns: {total_cols}")
    
    # Check for required new columns
    required_cols = [
        'IS_DELAYED', 'dep_3hour_col', 'arr_3hour_col',
        'flight_id', 'daily_flights', 'volume_zscore',
        'is_rare', 'is_abnormal_num', 'is_peak_day', 'is_extra_candidate'
    ]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if not missing_cols:
        print(f"   ✓ All {len(required_cols)} required feature columns present")
    else:
        print(f"   ⚠ Missing columns: {missing_cols}")
    
    # Check columns that should be dropped
    dropped_cols = ['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE', 
                   'source_file', 'is_delayed_15', 'processing_timestamp', 'sort_time']
    found_dropped = [c for c in dropped_cols if c in df.columns]
    
    if not found_dropped:
        print(f"   ✓ All unnecessary columns properly dropped")
    else:
        print(f"   ⚠ Found columns that should be dropped: {found_dropped}")
    
    # 3. IS_DELAYED distribution
    print(f"\n3. IS_DELAYED Distribution:")
    delay_dist = df.groupBy("IS_DELAYED").count().orderBy("IS_DELAYED").collect()
    
    for row in delay_dist:
        delay_val = row["IS_DELAYED"]
        count = row["count"]
        pct = (count / total_rows) * 100
        category_names = {
            0: "Category 0 (< 60 min)",
            1: "Category 1 (60-119 min)",
            2: "Category 2 (>= 120 min)"
        }
        label = category_names.get(delay_val, f"Category {delay_val}")
        print(f"   {label:<25} {count:>8,} ({pct:>5.2f}%)")
        validation_results[f'delay_cat_{delay_val}'] = count
    
    # 4. 3-hour bucket validation
    print(f"\n4. 3-Hour Time Buckets:")
    dep_buckets = df.select("dep_3hour_col").distinct().count()
    arr_buckets = df.select("arr_3hour_col").distinct().count()
    print(f"   dep_3hour_col unique values: {dep_buckets}")
    print(f"   arr_3hour_col unique values: {arr_buckets}")
    
    # Show sample buckets
    print(f"   Sample dep_3hour_col values:")
    sample_buckets = df.select("dep_3hour_col").distinct().orderBy("dep_3hour_col").limit(8).collect()
    for row in sample_buckets:
        print(f"     - {row['dep_3hour_col']}")
    
    # 5. Flight volume features
    print(f"\n5. Flight Volume Features:")
    
    rare_count = df.filter(col("is_rare") == 1).count()
    abnormal_count = df.filter(col("is_abnormal_num") == 1).count()
    peak_count = df.filter(col("is_peak_day") == 1).count()
    extra_count = df.filter(col("is_extra_candidate") == 1).count()
    
    print(f"   is_rare flights:         {rare_count:>8,} ({rare_count/total_rows*100:>5.2f}%)")
    print(f"   is_abnormal_num flights: {abnormal_count:>8,} ({abnormal_count/total_rows*100:>5.2f}%)")
    print(f"   is_peak_day flights:     {peak_count:>8,} ({peak_count/total_rows*100:>5.2f}%)")
    print(f"   is_extra_candidate:      {extra_count:>8,} ({extra_count/total_rows*100:>5.2f}%)")
    
    # 6. Data quality checks
    print(f"\n6. Data Quality:")
    
    # Check for nulls in key columns
    null_checks = ['IS_DELAYED', 'dep_3hour_col', 'arr_3hour_col', 'flight_id']
    for col_name in null_checks:
        null_count = df.filter(col(col_name).isNull()).count()
        if null_count == 0:
            print(f"   ✓ {col_name}: no nulls")
        else:
            print(f"   ⚠ {col_name}: {null_count:,} nulls ({null_count/total_rows*100:.2f}%)")
    
    validation_results['schema_valid'] = True
    
    print(f"\n{'='*80}")
    print("FLIGHT GOLD VALIDATION COMPLETE")
    print("="*80)
    
    return validation_results


def main(snapshotdate=None):
    """
    Main execution function
    
    Args:
        snapshotdate: Optional date string "YYYY-MM-DD" for daily OOT processing
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FLIGHT DELAY PREDICTION - FLIGHT GOLD LAYER PROCESSING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine processing mode
    if snapshotdate:
        print(f"Mode: DAILY OOT")
        print(f"Snapshot Date: {snapshotdate}")
    else:
        print(f"Mode: HISTORICAL BATCH")
    
    print("="*80 + "\n")
    
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    if snapshotdate:
        # OOT mode
        snap_tag = snapshotdate.replace("-", "_")
        silver_path = f"../datamart/silver/flight/silver_flight_oot_{snap_tag}.parquet"
        flight_gold_output_path = f"../datamart/gold/flight/flight_gold_oot_{snap_tag}.parquet"
        is_daily = True
    else:
        # Historical mode
        silver_path = "../datamart/silver/flight/silver_flight_historical.parquet"
        flight_gold_output_path = "../datamart/gold/flight/flight_gold_historical.parquet"
        is_daily = False
    
    # Check if silver directory exists
    if not os.path.exists(silver_path):
        print(f"ERROR: Silver data not found: {silver_path}")
        print("\nPlease run Silver processing first:")
        if snapshotdate:
            print(f"  python silver_flight_store.py --snapshotdate {snapshotdate}")
        else:
            print(f"  python silver_flight_store.py")
        sys.exit(1)
        return

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelayGold") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")

    try:
        # Process to Flight Gold
        df_gold = process_silver_to_flight_gold(
            silver_path=silver_path,
            flight_gold_output_path=flight_gold_output_path,
            spark=spark,
            snapshot_date=snapshotdate,
            output_mode='overwrite'
        )
        
        # Validate outputs
        validation_results = validate_flight_gold_parquet(
            flight_gold_output_path=flight_gold_output_path,
            spark=spark,
            is_daily=is_daily
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Flight Gold Parquet created: {flight_gold_output_path}")
        print(f"✓ Total rows: {validation_results.get('total_rows', 'N/A'):,}")
        print(f"✓ Total columns: {validation_results.get('total_columns', 'N/A')}")
        print(f"✓ Processing time: {elapsed_time/60:.1f} minutes")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if snapshotdate:
            print(f"\n✓ Daily Flight Gold created for {snapshotdate}")
            print("\nNext:")
            print("1. Join with weather to create Gold Combined")
            print(f"   python gold_combined_store.py --snapshotdate {snapshotdate}")
        else:
            print("\n1. Inspect Flight Gold Parquet file:")
            print(f"   ls -lh {flight_gold_output_path}/")
            print("\n2. Next: Join with weather data")
            print("   python gold_combined_store.py")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print("="*80)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flight Gold Layer Processing for Flight Delay Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 24 months (historical batch)
  python gold_flight_store.py
  
  # Process single day for OOT inference
  python gold_flight_store.py --snapshotdate 2025-01-15
        """
    )
    
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=False,
        default=None,
        help="Snapshot date for daily OOT processing (format: YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    # Validate date format if provided
    if args.snapshotdate:
        try:
            datetime.strptime(args.snapshotdate, "%Y-%m-%d")
        except ValueError:
            print(f"ERROR: Invalid date format: {args.snapshotdate}")
            print("Expected format: YYYY-MM-DD (e.g., 2025-01-15)")
            exit(1)
    
    main(snapshotdate=args.snapshotdate)


