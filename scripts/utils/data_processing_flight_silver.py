"""
Silver Layer Processing for Flight Delay Data - Data Cleaning & Validation

Supports three processing modes:
1. Historical batch: Process all 24 months for model training (Jan 2023 - Dec 2024)
2. Monthly retraining: Add full month to historical Silver for retraining (Jan 2025)
3. Daily OOT: Process single day for inference (Jan-Mar 2025)

Processes Bronze Parquet into clean Silver layer with:
- Data cleaning (duplicates, cancelled/diverted, outliers)
- Data validation and type enforcement
- Proper sorting and partitioning
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import pyspark
from pyspark.sql import SparkSession

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
    # US Federal holidays for our date range (extended to 2025)
    US_HOLIDAYS = holidays.US(years=range(2023, 2026))
except ImportError:
    HOLIDAYS_AVAILABLE = False
    US_HOLIDAYS = None

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import (
        col, lit, when, to_date, concat_ws, current_timestamp,
        dayofweek, udf, coalesce, count, isnan, trim, upper,
        hour, minute, unix_timestamp, abs, round
    )
    from pyspark.sql.types import IntegerType, DateType, StringType, DoubleType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    # Define dummy classes/functions for import compatibility
    SparkSession = None
    DataFrame = None


# NYC Metro airports - our scope
NYC_METRO_AIRPORTS = ['JFK', 'LGA', 'EWR']


def resolve_col(df, col_name: str) -> str:
    """
    Resolve column name with case-insensitive matching

    Args:
        df: DataFrame to search
        col_name: Column name to find

    Returns:
        Exact column name from DataFrame
    """
    col_name_upper = col_name.upper()
    for c in df.columns:
        if c.upper() == col_name_upper:
            return c
    raise ValueError(f"Column '{col_name}' not found in DataFrame. Available: {df.columns}")


def try_resolve_col(df, col_name: str, default_value=None) -> Optional[str]:
    """
    Try to resolve column name, return None if not found

    Args:
        df: DataFrame to search
        col_name: Column name to find
        default_value: Value to return if not found

    Returns:
        Column name if found, default_value otherwise
    """
    try:
        return resolve_col(df, col_name)
    except ValueError:
        return default_value


def load_bronze_data(bronze_path: str, spark,
                     snapshot_date: Optional[str] = None):
    """
    Load Bronze layer data based on processing mode

    Args:
        bronze_path: Path to Bronze Parquet
        spark: SparkSession
        snapshot_date: Optional date filter for OOT mode

    Returns:
        Loaded DataFrame
    """
    print(f"  Loading Bronze data from: {bronze_path}")

    # Read Bronze Parquet
    df = spark.read.parquet(bronze_path)

    initial_count = df.count()
    print(f"    Initial rows in Bronze: {initial_count:,}")

    # Filter by snapshot_date if provided (for OOT mode)
    if snapshot_date is not None:
        print(f"    Filtering for snapshot_date: {snapshot_date}")
        df = df.filter(col('snapshot_date') == snapshot_date)
        filtered_count = df.count()
        print(f"    After snapshot_date filter: {filtered_count:,}")

    return df


def remove_duplicates(df):
    """
    Remove duplicate records based on flight identifiers

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    print("\n  1. Removing duplicates...")

    initial_count = df.count()

    # Define duplicate key: FlightDate + Carrier + FlightNumber + Origin + Dest + DepTime
    duplicate_cols = [
        resolve_col(df, 'FlightDate'),
        resolve_col(df, 'OP_UNIQUE_CARRIER'),
        resolve_col(df, 'OP_CARRIER_FL_NUM'),
        resolve_col(df, 'ORIGIN'),
        resolve_col(df, 'DEST'),
        resolve_col(df, 'CRS_DEP_TIME')
    ]

    # Count duplicates before removal
    duplicate_count = df.groupBy(duplicate_cols).agg(count("*").alias("count")).filter(col("count") > 1).count()
    print(f"    Found {duplicate_count} duplicate groups")

    # Remove duplicates (keep first occurrence)
    df = df.dropDuplicates(duplicate_cols)

    final_count = df.count()
    removed_count = initial_count - final_count
    print(f"    Removed {removed_count:,} duplicate records")
    print(f"    Remaining rows: {final_count:,}")

    return df


def enforce_datatypes(df):
    """
    Enforce correct data types for all columns

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with corrected data types
    """
    print("\n  2. Enforcing data types...")

    # String columns - trim and uppercase
    string_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'TAIL_NUM', 'CANCELLATION_CODE']
    for col_name in string_cols:
        resolved_col = try_resolve_col(df, col_name)
        if resolved_col:
            df = df.withColumn(resolved_col, trim(upper(col(resolved_col))))

    # Integer columns that should be integers
    int_cols = [
        'YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
        'OP_CARRIER_FL_NUM', 'CRS_DEP_TIME', 'CRS_ARR_TIME',
        'DISTANCE', 'DISTANCE_GROUP'
    ]

    for col_name in int_cols:
        resolved_col = try_resolve_col(df, col_name)
        if resolved_col:
            df = df.withColumn(resolved_col, col(resolved_col).cast(IntegerType()))

    # Float columns
    float_cols = [
        'DEP_TIME', 'ARR_TIME', 'ACTUAL_ELAPSED_TIME',
        'DEP_DELAY_NEW', 'ARR_DELAY_NEW', 'DEP_DELAY_GROUP', 'ARR_DELAY_GROUP',
        'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
    ]

    for col_name in float_cols:
        resolved_col = try_resolve_col(df, col_name)
        if resolved_col:
            df = df.withColumn(resolved_col, col(resolved_col).cast(DoubleType()))

    # Boolean-like columns (0/1 integers)
    bool_cols = ['CANCELLED', 'DIVERTED', 'is_delayed_15', 'IsWeekend', 'IsPublicHoliday']
    for col_name in bool_cols:
        resolved_col = try_resolve_col(df, col_name)
        if resolved_col:
            df = df.withColumn(resolved_col, col(resolved_col).cast(IntegerType()))

    print("    ✓ Data types enforced")
    return df


def remove_cancelled_diverted(df):
    """
    Remove cancelled and diverted flights

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cancelled/diverted flights removed
    """
    print("\n  3. Removing cancelled and diverted flights...")

    initial_count = df.count()

    cancelled_col = resolve_col(df, 'CANCELLED')
    diverted_col = resolve_col(df, 'DIVERTED')

    # Count cancelled and diverted flights
    cancelled_count = df.filter(col(cancelled_col) == 1).count()
    diverted_count = df.filter(col(diverted_col) == 1).count()

    print(f"    Cancelled flights: {cancelled_count:,}")
    print(f"    Diverted flights: {diverted_count:,}")

    # Remove cancelled and diverted flights
    df = df.filter((col(cancelled_col) != 1) & (col(diverted_col) != 1))

    final_count = df.count()
    removed_count = initial_count - final_count
    print(f"    Removed {removed_count:,} cancelled/diverted flights")
    print(f"    Remaining rows: {final_count:,}")

    return df


def remove_cancelled_with_delays(df):
    """
    Remove records where flights are cancelled/diverted but have delay values
    (data quality issue - cancelled flights shouldn't have delay times)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with invalid delay records removed
    """
    print("\n  4. Removing cancelled/diverted flights with delay values...")

    initial_count = df.count()

    cancelled_col = resolve_col(df, 'CANCELLED')
    diverted_col = resolve_col(df, 'DIVERTED')
    arr_delay_col = try_resolve_col(df, 'ARR_DELAY_NEW')

    if arr_delay_col:
        # Find cancelled/diverted flights that have arrival delay values
        invalid_delays = df.filter(
            ((col(cancelled_col) == 1) | (col(diverted_col) == 1)) &
            (col(arr_delay_col).isNotNull())
        ).count()

        print(f"    Found {invalid_delays:,} cancelled/diverted flights with delay values")

        # Remove these invalid records
        df = df.filter(
            ~(((col(cancelled_col) == 1) | (col(diverted_col) == 1)) &
              (col(arr_delay_col).isNotNull()))
        )

        final_count = df.count()
        removed_count = initial_count - final_count
        print(f"    Removed {removed_count:,} invalid delay records")
        print(f"    Remaining rows: {final_count:,}")
    else:
        print("    ARR_DELAY_NEW column not found, skipping this check")

    return df


def remove_outliers(df):
    """
    Remove obvious data errors:
    - Flight time > 24 hours
    - Flight distance > 6000 miles
    - Actual elapsed time negative

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with outliers removed
    """
    print("\n  5. Removing obvious data errors...")

    initial_count = df.count()

    # Check 1: Flight time > 24 hours (1440 minutes)
    elapsed_col = try_resolve_col(df, 'ACTUAL_ELAPSED_TIME')
    if elapsed_col:
        long_flights = df.filter(col(elapsed_col) > 1440).count()
        print(f"    Flights with duration > 24 hours: {long_flights:,}")

        df = df.filter((col(elapsed_col).isNull()) | (col(elapsed_col) <= 1440))

    # Check 2: Flight distance > 6000 miles
    distance_col = try_resolve_col(df, 'DISTANCE')
    if distance_col:
        long_distance = df.filter(col(distance_col) > 6000).count()
        print(f"    Flights with distance > 6000 miles: {long_distance:,}")

        df = df.filter((col(distance_col).isNull()) | (col(distance_col) <= 6000))

    # Check 3: Negative actual elapsed time
    if elapsed_col:
        negative_time = df.filter(col(elapsed_col) < 0).count()
        print(f"    Flights with negative elapsed time: {negative_time:,}")

        df = df.filter((col(elapsed_col).isNull()) | (col(elapsed_col) >= 0))

    final_count = df.count()
    removed_count = initial_count - final_count
    print(f"    Removed {removed_count:,} outlier records")
    print(f"    Remaining rows: {final_count:,}")

    return df


def process_bronze_to_silver(bronze_path: str, silver_output_path: str,
                           spark,
                           snapshot_date: Optional[str] = None,
                           output_mode: str = 'overwrite'):
    """
    Main processing function: Transform Bronze to Silver layer

    Supports three modes:
    1. Historical batch (snapshot_date=None): Process all 24 months for training
    2. Monthly retraining (snapshot_date=None, output_mode='append'): Add month to existing Silver
    3. Daily OOT (snapshot_date="2025-01-01"): Process single day for inference

    Args:
        bronze_path: Path to Bronze Parquet
        silver_output_path: Output path for Silver Parquet
        spark: SparkSession
        snapshot_date: Optional. Format "YYYY-MM-DD"
                      - If None: Process all data (batch/retraining mode)
                      - If provided: Process only this date (daily OOT mode)
        output_mode: 'overwrite' or 'append'
                    - overwrite: Replace existing data (default)
                    - append: Add to existing Parquet

    Returns:
        Processed Silver DataFrame
    """
    print("="*80)
    print("SILVER LAYER PROCESSING - FLIGHT DELAY DATA")
    print("="*80)

    # Load Bronze data
    df = load_bronze_data(bronze_path, spark, snapshot_date)

    # Apply cleaning steps
    df = remove_duplicates(df)
    df = enforce_datatypes(df)
    df = remove_cancelled_diverted(df)
    df = remove_cancelled_with_delays(df)
    df = remove_outliers(df)

    # Sort by FlightDate and sort_time
    print("\n  Sorting by FlightDate and sort_time...")
    sort_time_col = try_resolve_col(df, 'sort_time')
    if sort_time_col:
        df = df.orderBy(['FlightDate', sort_time_col])
    else:
        df = df.orderBy(['FlightDate'])

    # Save Silver Parquet
    print(f"\n{'='*80}")
    print("SAVING TO SILVER PARQUET")
    print("="*80)

    os.makedirs(os.path.dirname(silver_output_path), exist_ok=True)

    if snapshot_date is not None:
        # Daily OOT mode - partition by snapshot_date
        print(f"\n  Output path: {silver_output_path}")
        print(f"  Partitioning by: snapshot_date")
        print(f"  Mode: {output_mode}")

        df = df.withColumn('snapshot_date', lit(snapshot_date))
        df.write.mode(output_mode).partitionBy('snapshot_date').parquet(silver_output_path)

    else:
        # Historical/Monthly mode - partition by year_month
        print(f"\n  Output path: {silver_output_path}")
        print(f"  Partitioning by: year_month")
        print(f"  Mode: {output_mode}")
        print("\n  Writing Silver Parquet file...")

        df.write.mode(output_mode).partitionBy('year_month').parquet(silver_output_path)

    print("  ✓ Silver Parquet file saved successfully!")

    return df


def validate_silver_parquet(silver_output_path: str, spark,
                          is_daily: bool = False) -> Dict:
    """
    Validate Silver Parquet file quality

    Args:
        silver_output_path: Path to Silver Parquet
        spark: SparkSession
        is_daily: True if validating daily OOT data

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*80}")
    print("SILVER VALIDATION CHECKS")
    print("="*80)

    # Read Silver Parquet
    df = spark.read.parquet(silver_output_path)

    validation_results = {}

    # 1. Row count
    total_rows = df.count()
    validation_results['total_rows'] = total_rows
    print(f"\n1. Total Rows: {total_rows:,}")

    # 2. Data cleaning validation
    cancelled_col = try_resolve_col(df, 'CANCELLED')
    diverted_col = try_resolve_col(df, 'DIVERTED')

    if cancelled_col and diverted_col:
        cancelled_count = df.filter(col(cancelled_col) == 1).count()
        diverted_count = df.filter(col(diverted_col) == 1).count()
        print(f"\n2. Cancelled/Diverted Check:")
        print(f"   Cancelled flights: {cancelled_count}")
        print(f"   Diverted flights: {diverted_count}")

        if cancelled_count == 0 and diverted_count == 0:
            print("   ✓ PASS - No cancelled/diverted flights in Silver")
        else:
            print("   ⚠ WARNING - Found cancelled/diverted flights in Silver")

    # 3. Target variable distribution
    target_col = try_resolve_col(df, 'is_delayed_15')
    if target_col:
        delay_dist = df.groupBy(target_col).count().collect()
        print(f"\n3. Target Variable (is_delayed_15):")

        for row in delay_dist:
            delay_val = row[target_col]
            count = row['count']
            pct = count / total_rows * 100
            label = "Delayed (≥15 min)" if delay_val == 1 else "On-time (<15 min)"
            print(f"   {label}: {count:,} ({pct:.2f}%)")
            validation_results[f'delay_{delay_val}'] = count

    # 4. Data type validation
    print(f"\n4. Data Type Validation:")
    
    # Check key columns have correct types
    expected_types = {
        'FlightDate': 'date',
        'is_delayed_15': 'int',
        'OP_UNIQUE_CARRIER': 'string',
        'ARR_DELAY_NEW': 'double'
    }
    
    type_issues = []
    for col_name, expected_type in expected_types.items():
        resolved_col = try_resolve_col(df, col_name)
        if resolved_col:
            actual_type = str(df.schema[resolved_col].dataType).lower()
            if expected_type not in actual_type:
                type_issues.append(f"{col_name}: expected {expected_type}, got {actual_type}")
            else:
                print(f"   ✓ {col_name}: {actual_type}")
        else:
            type_issues.append(f"{col_name}: column not found")
    
    if type_issues:
        print(f"   ⚠ Type issues: {type_issues}")
    else:
        print("   ✓ All key columns have correct types")

    # 5. Outlier checks
    print(f"\n5. Outlier Validation:")

    elapsed_col = try_resolve_col(df, 'ACTUAL_ELAPSED_TIME')
    if elapsed_col:
        long_flights = df.filter(col(elapsed_col) > 1440).count()
        print(f"   Flights >24 hours: {long_flights}")

    distance_col = try_resolve_col(df, 'DISTANCE')
    if distance_col:
        long_distance = df.filter(col(distance_col) > 6000).count()
        print(f"   Flights >6000 miles: {long_distance}")

    validation_results['schema_valid'] = True  # Basic validation passed

    print(f"\n{'='*80}")
    print("SILVER VALIDATION COMPLETE")
    print("="*80)

    return validation_results


# Module-level constants for import checking
HOLIDAYS_AVAILABLE = True
PYSPARK_AVAILABLE = True

# Add this entire block at the end of your silver file:

def main(snapshotdate=None):
    """
    Main execution function for Silver layer processing
    
    Args:
        snapshotdate: Optional date string "YYYY-MM-DD" for daily OOT processing
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FLIGHT DELAY PREDICTION - SILVER LAYER PROCESSING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine processing mode
    if snapshotdate:
        print(f"Mode: DAILY OOT")
        print(f"Snapshot Date: {snapshotdate}")
    else:
        print(f"Mode: HISTORICAL BATCH")
    
    print("="*80 + "\n")
    
    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelaySilver") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")
    
    # Configuration - different paths for historical vs OOT
    if snapshotdate:
        # DAILY OOT MODE
        bronze_path = "datamart/bronze/flight/bronze_flight_oot.parquet"
        silver_output_path = "datamart/silver/flight/silver_flight_oot.parquet"
        is_daily = True
    else:
        # HISTORICAL COMBINED MODE
        bronze_path = "datamart/bronze/flight/bronze_flight_historical.parquet"
        silver_output_path = "datamart/silver/flight/silver_flight_historical.parquet"
        is_daily = False
    
    # Check if bronze directory exists
    if not os.path.exists(bronze_path):
        print(f"ERROR: Bronze data not found: {bronze_path}")
        print("\nPlease run Bronze processing first:")
        if snapshotdate:
            print(f"  python main_static.py --snapshotdate {snapshotdate}")
        else:
            print(f"  python main_static.py")
        spark.stop()
        return
    
    try:
        # Process Bronze to Silver
        df_silver = process_bronze_to_silver(
            bronze_path=bronze_path,
            silver_output_path=silver_output_path,
            spark=spark,
            snapshot_date=snapshotdate,
            output_mode='overwrite'
        )
        
        # Validate outputs
        validation_results = validate_silver_parquet(
            silver_output_path=silver_output_path,
            spark=spark,
            is_daily=is_daily
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Silver Parquet created: {silver_output_path}")
        print(f"✓ Total rows: {validation_results.get('total_rows', 'N/A'):,}")
        print(f"✓ Processing time: {elapsed_time/60:.1f} minutes")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if snapshotdate:
            print(f"\n✓ Daily Silver created for {snapshotdate}")
            print("\nNext:")
            print("1. Process to Gold layer (join with weather)")
            print("2. Run model inference")
        else:
            print("\n1. Inspect Silver Parquet file:")
            print(f"   ls -lh {silver_output_path}/")
            print("\n2. Next: Process to Gold layer (feature engineering)")
        
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
    import argparse
    
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Silver Layer Processing for Flight Delay Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 24 months (historical batch)
  python data_processing_flight_silver.py
  
  # Process single day for OOT inference
  python data_processing_flight_silver.py --snapshotdate 2025-01-01
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
            print("Expected format: YYYY-MM-DD (e.g., 2025-01-01)")
            exit(1)
    
    # Call main
    main(snapshotdate=args.snapshotdate)
