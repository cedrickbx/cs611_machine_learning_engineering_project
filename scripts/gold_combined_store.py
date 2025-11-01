"""
Main Script: Gold Combined Layer Processing
Join Flight Gold with Weather data to create final Gold Combined layer:
1. Historical batch processing (24 months)
2. Daily OOT processing (single day)

Usage:
    # Historical batch (default)
    python gold_combined_store.py
    
    # Daily OOT processing  
    python gold_combined_store.py --snapshotdate 2025-01-15

Output:
    Historical: datamart/gold/combined/gold_combined_historical.parquet/
    Daily OOT:  datamart/gold/combined/gold_combined_oot_YYYY_MM_DD.parquet/
"""

import os
import time
import argparse
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, substring, to_date, broadcast
import sys


# NYC Metro airports
NYC_METRO_AIRPORTS = ['JFK', 'LGA', 'EWR']


def map_3hour_to_weather_time(time_3hour_col):
    """
    Map 3-hour bucket to weather time format
    
    Examples:
        "0000-0259" → "0000"
        "0300-0559" → "0300"
        "0600-0859" → "0600"
    
    Args:
        time_3hour_col: 3-hour bucket string (e.g., "0600-0859")
        
    Returns:
        Weather time string (e.g., "0600")
    """
    if time_3hour_col is None:
        return None
    # Extract first 4 characters (start time of bucket)
    return time_3hour_col[:4]


def prepare_weather_data(weather_path, spark, snapshot_date=None):
    """
    Load and prepare weather data for joining
    
    Args:
        weather_path: Path to silver_weather_store parquet
        spark: SparkSession
        snapshot_date: Optional date filter for OOT mode
        
    Returns:
        Weather DataFrame with standardized columns
    """
    print(f"\n  Loading weather data from: {weather_path}")
    
    df_weather = spark.read.parquet(weather_path)
    
    initial_count = df_weather.count()
    print(f"    Initial weather rows: {initial_count:,}")
    
    # Convert date column to proper date type if it's string
    if dict(df_weather.dtypes).get('date') == 'string':
        df_weather = df_weather.withColumn('date', to_date(col('date')))
    
    # Filter by snapshot_date if provided (OOT mode)
    if snapshot_date:
        print(f"    Filtering weather for date: {snapshot_date}")
        df_weather = df_weather.filter(col('date') == snapshot_date)
        filtered_count = df_weather.count()
        print(f"    After date filter: {filtered_count:,} rows")
    
    # Rename columns for clarity in joined dataset
    # Prefix weather columns with 'weather_' except for join keys
    weather_feature_cols = [
        'wind_dir_deg', 'wind_type', 'wind_speed_mps', 
        'ceiling_m', 'ceiling_code', 'visibility_m', 
        'temp_c', 'dewpoint_c', 'slp_hpa',
        'visibility_var_code', 'visibility_var_code_idx'
    ]
    
    for col_name in weather_feature_cols:
        if col_name in df_weather.columns:
            df_weather = df_weather.withColumnRenamed(col_name, f'weather_{col_name}')
    
    # Rename 'name' to 'airport' for clarity
    if 'name' in df_weather.columns:
        df_weather = df_weather.withColumnRenamed('name', 'weather_airport')
    
    # time is already in format "0000", "0300", etc.
    df_weather = df_weather.withColumnRenamed('time', 'weather_time')
    df_weather = df_weather.withColumnRenamed('date', 'weather_date')
    
    print(f"    ✓ Weather data prepared: {df_weather.count():,} rows")
    
    return df_weather


def prepare_flight_data(flight_gold_path, spark, snapshot_date=None, is_daily=False):
    """
    Load and prepare flight gold data for joining
    
    Args:
        flight_gold_path: Path to flight_gold parquet
        spark: SparkSession
        snapshot_date: Optional date filter
        is_daily: If True, drop FlightDate column after processing (OOT mode)
        
    Returns:
        Flight DataFrame ready for joining
    """
    print(f"\n  Loading flight gold data from: {flight_gold_path}")
    
    df_flight = spark.read.parquet(flight_gold_path)
    
    # Filter by snapshot_date if provided
    if snapshot_date:
        print(f"    Filtering for snapshot_date: {snapshot_date}")
        df_flight = df_flight.filter(col('snapshot_date') == snapshot_date)
    
    initial_count = df_flight.count()
    initial_cols = len(df_flight.columns)
    print(f"    Initial flight rows: {initial_count:,}")
    print(f"    Initial flight columns: {initial_cols}")
    
    # Map 3-hour buckets to weather time format
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    map_time_udf = udf(map_3hour_to_weather_time, StringType())
    
    df_flight = df_flight.withColumn('dep_weather_time', map_time_udf(col('dep_3hour_col')))
    df_flight = df_flight.withColumn('arr_weather_time', map_time_udf(col('arr_3hour_col')))
    
    # Convert FlightDate to date type if needed
    if 'FlightDate' in df_flight.columns:
        if dict(df_flight.dtypes).get('FlightDate') == 'string':
            df_flight = df_flight.withColumn('FlightDate', to_date(col('FlightDate')))
    
    print(f"    ✓ Mapped 3-hour buckets to weather time format")
    
    return df_flight


def join_flight_with_weather(df_flight, df_weather, spark, is_daily=False):
    """
    Join flight data with weather data based on time buckets and airports
    
    Join Strategy:
    - For departing NYC flights (ORIGIN in NYC): join on ORIGIN, dep_weather_time, FlightDate
    - For arriving NYC flights (DEST in NYC): join on DEST, arr_weather_time, FlightDate
    - Weather data is broadcast (small dataset) to all matching flights
    
    Args:
        df_flight: Flight Gold DataFrame
        df_weather: Weather DataFrame
        spark: SparkSession
        is_daily: If True, OOT mode (all flights same date)
        
    Returns:
        Joined DataFrame with flight + weather features
    """
    print(f"\n  Joining flight data with weather...")
    
    initial_count = df_flight.count()
    
    # Separate flights into departing and arriving from/to NYC
    df_departing = df_flight.filter(col('ORIGIN').isin(NYC_METRO_AIRPORTS))
    df_arriving = df_flight.filter(col('DEST').isin(NYC_METRO_AIRPORTS))
    
    departing_count = df_departing.count()
    arriving_count = df_arriving.count()
    
    print(f"    Flights departing from NYC: {departing_count:,}")
    print(f"    Flights arriving to NYC: {arriving_count:,}")
    
    # Broadcast weather data (small dataset ~24 rows per day)
    df_weather_broadcast = broadcast(df_weather)
    
    # Join departing flights with weather
    # Join on: FlightDate, ORIGIN airport, departure time bucket
    print(f"\n    Joining departing flights with weather...")
    
    if is_daily:
        # OOT mode: all flights same date, just match on airport and time
        df_departing_joined = df_departing.join(
            df_weather_broadcast,
            (df_departing['ORIGIN'] == df_weather_broadcast['weather_airport']) &
            (df_departing['dep_weather_time'] == df_weather_broadcast['weather_time']),
            how='left'
        )
    else:
        # Historical mode: need to match on date too
        df_departing_joined = df_departing.join(
            df_weather_broadcast,
            (df_departing['FlightDate'] == df_weather_broadcast['weather_date']) &
            (df_departing['ORIGIN'] == df_weather_broadcast['weather_airport']) &
            (df_departing['dep_weather_time'] == df_weather_broadcast['weather_time']),
            how='left'
        )
    
    # Drop weather metadata columns (keep only features)
    weather_meta_cols = ['weather_airport', 'weather_date', 'weather_time', 'T+1_forecast']
    df_departing_joined = df_departing_joined.drop(*[c for c in weather_meta_cols if c in df_departing_joined.columns])
    
    departing_joined_count = df_departing_joined.count()
    print(f"    ✓ Departing flights joined: {departing_joined_count:,}")
    
    # Join arriving flights with weather
    print(f"\n    Joining arriving flights with weather...")
    
    if is_daily:
        df_arriving_joined = df_arriving.join(
            df_weather_broadcast,
            (df_arriving['DEST'] == df_weather_broadcast['weather_airport']) &
            (df_arriving['arr_weather_time'] == df_weather_broadcast['weather_time']),
            how='left'
        )
    else:
        df_arriving_joined = df_arriving.join(
            df_weather_broadcast,
            (df_arriving['FlightDate'] == df_weather_broadcast['weather_date']) &
            (df_arriving['DEST'] == df_weather_broadcast['weather_airport']) &
            (df_arriving['arr_weather_time'] == df_weather_broadcast['weather_time']),
            how='left'
        )
    
    df_arriving_joined = df_arriving_joined.drop(*[c for c in weather_meta_cols if c in df_arriving_joined.columns])
    
    arriving_joined_count = df_arriving_joined.count()
    print(f"    ✓ Arriving flights joined: {arriving_joined_count:,}")
    
    # Union departing and arriving flights
    print(f"\n    Combining departing and arriving flights...")
    df_combined = df_departing_joined.union(df_arriving_joined)
    
    final_count = df_combined.count()
    print(f"    ✓ Combined flights: {final_count:,}")
    
    # Validate row count matches
    if final_count != initial_count:
        print(f"    ⚠ WARNING: Row count mismatch! Initial: {initial_count:,}, Final: {final_count:,}")
    else:
        print(f"    ✓ Row count preserved")
    
    # Check for null weather values
    weather_cols = [c for c in df_combined.columns if c.startswith('weather_')]
    if weather_cols:
        sample_weather_col = weather_cols[0]
        null_count = df_combined.filter(col(sample_weather_col).isNull()).count()
        null_pct = (null_count / final_count * 100) if final_count > 0 else 0
        
        if null_count > 0:
            print(f"    ⚠ Found {null_count:,} rows ({null_pct:.2f}%) with null weather data")
        else:
            print(f"    ✓ All rows have weather data")
    
    # Drop temporary join columns
    df_combined = df_combined.drop('dep_weather_time', 'arr_weather_time')
    
    return df_combined


def process_to_gold_combined(flight_gold_path, weather_path, gold_combined_output_path, 
                             spark, snapshot_date=None, output_mode='overwrite'):
    """
    Process Flight Gold + Weather to Gold Combined
    
    Args:
        flight_gold_path: Path to flight_gold parquet
        weather_path: Path to silver_weather_store parquet
        gold_combined_output_path: Output path
        spark: SparkSession
        snapshot_date: Optional date for OOT mode
        output_mode: Write mode
        
    Returns:
        Final Gold Combined DataFrame
    """
    print(f"\n{'='*80}")
    print("FLIGHT GOLD + WEATHER → GOLD COMBINED")
    print("="*80)
    
    is_daily = snapshot_date is not None
    
    # Load and prepare data
    df_weather = prepare_weather_data(weather_path, spark, snapshot_date)
    df_flight = prepare_flight_data(flight_gold_path, spark, snapshot_date, is_daily)
    
    # Join flight with weather
    df_combined = join_flight_with_weather(df_flight, df_weather, spark, is_daily)
    
    # If OOT mode, drop FlightDate column (all same date)
    if is_daily and 'FlightDate' in df_combined.columns:
        print(f"\n  Dropping FlightDate column (OOT mode - all flights same date)")
        df_combined = df_combined.drop('FlightDate')
    
    final_count = df_combined.count()
    final_cols = len(df_combined.columns)
    
    print(f"\n  Gold Combined Complete:")
    print(f"    Final rows: {final_count:,}")
    print(f"    Final columns: {final_cols}")
    
    # Save Gold Combined
    print(f"\n  Saving Gold Combined to: {gold_combined_output_path}")
    
    # Partition by snapshot_date
    df_combined.write.mode(output_mode).partitionBy("snapshot_date").parquet(gold_combined_output_path)
    
    print(f"  ✓ Gold Combined saved successfully")
    
    return df_combined


def validate_gold_combined_parquet(gold_combined_output_path, spark, is_daily=False):
    """
    Validate Gold Combined Parquet output
    
    Args:
        gold_combined_output_path: Path to Gold Combined Parquet
        spark: SparkSession
        is_daily: Whether this is daily OOT or historical batch
        
    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*80}")
    print("GOLD COMBINED VALIDATION CHECKS")
    print("="*80)
    
    df = spark.read.parquet(gold_combined_output_path)
    
    validation_results = {}
    
    # 1. Row count
    total_rows = df.count()
    validation_results['total_rows'] = total_rows
    print(f"\n1. Total Rows: {total_rows:,}")
    
    # 2. Column count
    total_cols = len(df.columns)
    validation_results['total_columns'] = total_cols
    print(f"\n2. Total Columns: {total_cols}")
    
    # Count feature types
    weather_cols = [c for c in df.columns if c.startswith('weather_')]
    flight_cols = [c for c in df.columns if not c.startswith('weather_') and c != 'snapshot_date']
    
    print(f"   Flight feature columns: {len(flight_cols)}")
    print(f"   Weather feature columns: {len(weather_cols)}")
    
    # 3. Check target variable
    print(f"\n3. Target Variable (IS_DELAYED):")
    if 'IS_DELAYED' in df.columns:
        delay_dist = df.groupBy('IS_DELAYED').count().orderBy('IS_DELAYED').collect()
        for row in delay_dist:
            category = row['IS_DELAYED']
            count = row['count']
            pct = (count / total_rows) * 100
            print(f"   Category {category}: {count:>8,} ({pct:>5.2f}%)")
        print("   ✓ IS_DELAYED present")
    else:
        print("   ⚠ IS_DELAYED column not found!")
    
    # 4. Weather data coverage
    print(f"\n4. Weather Data Coverage:")
    
    if weather_cols:
        sample_weather_col = weather_cols[0]
        null_count = df.filter(col(sample_weather_col).isNull()).count()
        non_null_count = total_rows - null_count
        coverage_pct = (non_null_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"   Rows with weather data: {non_null_count:,} ({coverage_pct:.2f}%)")
        
        if null_count > 0:
            null_pct = (null_count / total_rows * 100)
            print(f"   ⚠ Rows with null weather: {null_count:,} ({null_pct:.2f}%)")
        else:
            print(f"   ✓ All rows have weather data")
        
        # List weather features
        print(f"\n   Weather features included:")
        for col_name in sorted(weather_cols)[:10]:  # Show first 10
            print(f"     - {col_name}")
        if len(weather_cols) > 10:
            print(f"     ... and {len(weather_cols) - 10} more")
    else:
        print("   ⚠ No weather columns found!")
    
    # 5. Check FlightDate handling
    print(f"\n5. FlightDate Column:")
    if 'FlightDate' in df.columns:
        if is_daily:
            print("   ⚠ FlightDate present in OOT mode (should be dropped)")
        else:
            unique_dates = df.select('FlightDate').distinct().count()
            print(f"   ✓ FlightDate present: {unique_dates} unique dates")
    else:
        if is_daily:
            print("   ✓ FlightDate correctly dropped (OOT mode)")
        else:
            print("   ⚠ FlightDate missing in historical mode")
    
    # 6. Required feature columns
    print(f"\n6. Required Features:")
    
    required_features = [
        'IS_DELAYED', 'dep_3hour_col', 'arr_3hour_col',
        'flight_id', 'is_extra_candidate'
    ]
    
    missing_features = [f for f in required_features if f not in df.columns]
    
    if not missing_features:
        print(f"   ✓ All required features present")
    else:
        print(f"   ⚠ Missing features: {missing_features}")
    
    validation_results['schema_valid'] = len(missing_features) == 0 and len(weather_cols) > 0
    
    print(f"\n{'='*80}")
    print("GOLD COMBINED VALIDATION COMPLETE")
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
    print("FLIGHT DELAY PREDICTION - GOLD COMBINED LAYER PROCESSING")
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
        flight_gold_path = f"datamart/gold/flight/flight_gold_oot_{snap_tag}.parquet"
        weather_path = f"datamart/silver/weather/silver_weather_store_{snapshotdate}.parquet"
        gold_combined_output_path = f"datamart/gold/combined/gold_combined_oot_{snap_tag}.parquet"
        is_daily = True
    else:
        # Historical mode
        flight_gold_path = "datamart/gold/flight/flight_gold_historical.parquet"
        weather_path = "datamart/silver/weather/silver_weather_store_historical.parquet"
        gold_combined_output_path = "datamart/gold/combined/gold_combined_historical.parquet"
        is_daily = False
    
    # Check if flight gold exists
    if not os.path.exists(flight_gold_path):
        print(f"ERROR: Flight Gold data not found: {flight_gold_path}")
        print("\nPlease run Flight Gold processing first:")
        if snapshotdate:
            print(f"  python gold_flight_store.py --snapshotdate {snapshotdate}")
        else:
            print(f"  python gold_flight_store.py")
        sys.exit(1)
        return
    
    # Check if weather exists
    if not os.path.exists(weather_path):
        print(f"ERROR: Weather data not found: {weather_path}")
        print("\nPlease ensure weather data is available at:")
        print(f"  {weather_path}")
        sys.exit(1)
        return

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelayGoldCombined") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")

    try:
        # Process to Gold Combined
        df_combined = process_to_gold_combined(
            flight_gold_path=flight_gold_path,
            weather_path=weather_path,
            gold_combined_output_path=gold_combined_output_path,
            spark=spark,
            snapshot_date=snapshotdate,
            output_mode='overwrite'
        )
        
        # Validate outputs
        validation_results = validate_gold_combined_parquet(
            gold_combined_output_path=gold_combined_output_path,
            spark=spark,
            is_daily=is_daily
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Gold Combined Parquet created: {gold_combined_output_path}")
        print(f"✓ Total rows: {validation_results.get('total_rows', 'N/A'):,}")
        print(f"✓ Total columns: {validation_results.get('total_columns', 'N/A')}")
        print(f"✓ Processing time: {elapsed_time/60:.1f} minutes")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if snapshotdate:
            print(f"\n✓ Daily Gold Combined created for {snapshotdate}")
            print("\nReady for:")
            print("1. Model inference/prediction")
            print("2. Model evaluation and metrics")
        else:
            print("\n1. Inspect Gold Combined Parquet file:")
            print(f"   ls -lh {gold_combined_output_path}/")
            print("\n2. Ready for:")
            print("   - Model training")
            print("   - Feature analysis")
            print("   - Data exploration")
        
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
        description="Gold Combined Layer Processing for Flight Delay Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 24 months (historical batch)
  python gold_combined_store.py
  
  # Process single day for OOT inference
  python gold_combined_store.py --snapshotdate 2025-01-15
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
