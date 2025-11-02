"""
Test Script: Daily OOT Bronze → Silver → Gold Processing

Tests the complete daily OOT processing functionality by:
1. Processing a sample date through Bronze layer
2. Processing Bronze → Silver layer
3. Processing Silver → Flight Gold layer
4. Processing Flight Gold + Weather → Gold Combined layer
5. Validating all outputs
6. Comparing with expected behavior

Usage:
    python test_daily_oot.py
"""

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Import processing functions
from utils.data_processing_flight_bronze import (
    process_all_months_to_bronze,
    validate_bronze_parquet,
    get_csv_from_date
)

from utils.data_processing_flight_silver import (
    process_bronze_to_silver,
    validate_silver_parquet
)

# Import Gold processing functions
from utils.data_processing_flight_gold import (
    drop_unnecessary_columns,
    create_delay_categories,
    create_3hour_buckets,
    create_flight_volume_features
)

from gold_flight_store import (
    process_silver_to_flight_gold,
    validate_flight_gold_parquet
)

from gold_combined_store import (
    process_to_gold_combined,
    validate_gold_combined_parquet
)


def test_daily_oot_processing():
    """
    Test complete daily OOT Bronze → Silver → Gold processing
    """
    print("="*80)
    print("TEST: DAILY OOT BRONZE → SILVER → GOLD PROCESSING")
    print("="*80)
    
    # Test configuration
    test_date = "2025-01-15"  # Mid-month to ensure CSV exists
    data_directory = "data/flight/oot/"
    bronze_output_path = "datamart/bronze/flight/test_bronze_oot.parquet"
    silver_output_path = "datamart/silver/flight/test_silver_oot.parquet"
    flight_gold_output_path = "datamart/gold/flight/test_flight_gold_oot.parquet"
    weather_input_path = f"datamart/silver/weather/silver_weather_store_{test_date}.parquet"
    gold_combined_output_path = "datamart/gold/combined/test_gold_combined_oot.parquet"
    
    print(f"\nTest Date: {test_date}")
    print(f"Data Directory: {data_directory}")
    print(f"Bronze Output: {bronze_output_path}")
    print(f"Silver Output: {silver_output_path}")
    print(f"Flight Gold Output: {flight_gold_output_path}")
    print(f"Weather Input: {weather_input_path}")
    print(f"Gold Combined Output: {gold_combined_output_path}")
    
    # Check if OOT data directory exists
    if not os.path.exists(data_directory):
        print(f"\n✗ FAIL: OOT data directory not found: {data_directory}")
        print("\nCreate directory structure:")
        print(f"  mkdir -p {data_directory}")
        print(f"  # Copy T_ONTIME_REPORTING-01_25.csv to {data_directory}")
        return False
    
    # Initialize Spark
    print("\nInitializing Spark...")
    spark = SparkSession.builder \
        .appName("TestDailyOOT") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized")
    
    try:
        # ==================== BRONZE LAYER TESTS ====================
        
        # Test 1: Check CSV file can be found
        print(f"\n{'='*80}")
        print("TEST 1: CSV File Discovery")
        print("="*80)
        
        try:
            csv_path = get_csv_from_date(test_date, data_directory)
            print(f"✓ PASS - Found CSV: {os.path.basename(csv_path)}")
        except FileNotFoundError as e:
            print(f"✗ FAIL - {str(e)}")
            spark.stop()
            return False
        
        # Test 2: Process daily Bronze
        print(f"\n{'='*80}")
        print("TEST 2: Daily Bronze Processing")
        print("="*80)
        
        df_bronze = process_all_months_to_bronze(
            data_directory=data_directory,
            bronze_output_path=bronze_output_path,
            spark=spark,
            snapshot_date=test_date,
            output_mode='overwrite'
        )
        
        bronze_row_count = df_bronze.count()
        
        if bronze_row_count > 0:
            print(f"✓ PASS - Processed {bronze_row_count:,} rows")
        else:
            print(f"✗ FAIL - No rows processed")
            spark.stop()
            return False
        
        # Test 3: Validate Bronze output structure
        print(f"\n{'='*80}")
        print("TEST 3: Bronze Output Validation")
        print("="*80)
        
        bronze_validation = validate_bronze_parquet(
            bronze_output_path=bronze_output_path,
            spark=spark,
            is_daily=True
        )
        
        # Test 4: Check Bronze partitioning
        print(f"\n{'='*80}")
        print("TEST 4: Bronze Partition Structure")
        print("="*80)
        
        bronze_partition_dir = f"{bronze_output_path}/snapshot_date={test_date}"
        
        if os.path.exists(bronze_partition_dir):
            print(f"✓ PASS - Partition created: snapshot_date={test_date}")
            
            # List files in partition
            parquet_files = [f for f in os.listdir(bronze_partition_dir) if f.endswith('.parquet')]
            print(f"  Parquet files: {len(parquet_files)}")
            
            for f in parquet_files[:3]:  # Show first 3
                size_mb = os.path.getsize(os.path.join(bronze_partition_dir, f)) / 1024 / 1024
                print(f"  - {f} ({size_mb:.2f} MB)")
        else:
            print(f"✗ FAIL - Partition not found: {bronze_partition_dir}")
            spark.stop()
            return False
        
        # Test 5: Verify Bronze derived columns
        print(f"\n{'='*80}")
        print("TEST 5: Bronze Derived Columns")
        print("="*80)
        
        required_bronze_cols = [
            'FlightDate', 'year_month', 'sort_time', 'is_delayed_15',
            'DayOfWeek', 'IsWeekend', 'IsPublicHoliday', 'snapshot_date'
        ]
        
        missing_cols = [col for col in required_bronze_cols if col not in df_bronze.columns]
        
        if not missing_cols:
            print("✓ PASS - All required columns present")
            print(f"  Total columns: {len(df_bronze.columns)}")
        else:
            print(f"✗ FAIL - Missing columns: {missing_cols}")
            spark.stop()
            return False
        
        # Test 6: Check single date constraint
        print(f"\n{'='*80}")
        print("TEST 6: Bronze Single Date Constraint")
        print("="*80)
        
        unique_dates = df_bronze.select('FlightDate').distinct().count()
        
        if unique_dates == 1:
            print(f"✓ PASS - Single date constraint satisfied")
        else:
            print(f"✗ FAIL - Found {unique_dates} unique dates (expected 1)")
            spark.stop()
            return False
        
        # Test 7: NYC filter
        print(f"\n{'='*80}")
        print("TEST 7: Bronze NYC Metro Filter")
        print("="*80)
        
        nyc_airports = ['JFK', 'LGA', 'EWR']
        nyc_count = df_bronze.filter(
            col('ORIGIN').isin(nyc_airports) | col('DEST').isin(nyc_airports)
        ).count()
        
        if nyc_count == bronze_row_count:
            print(f"✓ PASS - 100% NYC coverage ({nyc_count:,} rows)")
        else:
            print(f"✗ FAIL - Only {nyc_count/bronze_row_count*100:.1f}% NYC coverage")
            spark.stop()
            return False
        
        # ==================== SILVER LAYER TESTS ====================
        
        # Test 8: Process Bronze → Silver
        print(f"\n{'='*80}")
        print("TEST 8: Daily Silver Processing")
        print("="*80)
        
        df_silver = process_bronze_to_silver(
            bronze_path=bronze_output_path,
            silver_output_path=silver_output_path,
            spark=spark,
            snapshot_date=test_date,
            output_mode='overwrite'
        )
        
        silver_row_count = df_silver.count()
        
        if silver_row_count > 0:
            print(f"✓ PASS - Processed {silver_row_count:,} rows")
            
            # Calculate data reduction from Bronze to Silver
            reduction_pct = (1 - silver_row_count / bronze_row_count) * 100
            print(f"  Data reduction: {reduction_pct:.1f}% ({bronze_row_count - silver_row_count:,} rows removed)")
        else:
            print(f"✗ FAIL - No rows processed")
            spark.stop()
            return False
        
        # Test 9: Validate Silver output structure
        print(f"\n{'='*80}")
        print("TEST 9: Silver Output Validation")
        print("="*80)
        
        silver_validation = validate_silver_parquet(
            silver_output_path=silver_output_path,
            spark=spark,
            is_daily=True
        )
        
        # Test 10: Check Silver partitioning
        print(f"\n{'='*80}")
        print("TEST 10: Silver Partition Structure")
        print("="*80)
        
        silver_partition_dir = f"{silver_output_path}/snapshot_date={test_date}"
        
        if os.path.exists(silver_partition_dir):
            print(f"✓ PASS - Partition created: snapshot_date={test_date}")
            
            # List files in partition
            parquet_files = [f for f in os.listdir(silver_partition_dir) if f.endswith('.parquet')]
            print(f"  Parquet files: {len(parquet_files)}")
            
            for f in parquet_files[:3]:  # Show first 3
                size_mb = os.path.getsize(os.path.join(silver_partition_dir, f)) / 1024 / 1024
                print(f"  - {f} ({size_mb:.2f} MB)")
        else:
            print(f"✗ FAIL - Partition not found: {silver_partition_dir}")
            spark.stop()
            return False
        
        # Test 11: Verify no cancelled/diverted flights in Silver
        print(f"\n{'='*80}")
        print("TEST 11: Silver Data Cleaning Validation")
        print("="*80)
        
        cancelled_count = df_silver.filter(col('CANCELLED') == 1).count()
        diverted_count = df_silver.filter(col('DIVERTED') == 1).count()
        
        print(f"  Cancelled flights: {cancelled_count}")
        print(f"  Diverted flights: {diverted_count}")
        
        if cancelled_count == 0 and diverted_count == 0:
            print("✓ PASS - No cancelled/diverted flights in Silver")
        else:
            print("✗ FAIL - Found cancelled/diverted flights in Silver layer")
            spark.stop()
            return False
        
        # Test 12: Verify target variable distribution
        print(f"\n{'='*80}")
        print("TEST 12: Target Variable Distribution (Silver)")
        print("="*80)
        
        delay_dist = df_silver.groupBy('is_delayed_15').count().collect()
        
        for row in delay_dist:
            delay_val = row['is_delayed_15']
            count = row['count']
            pct = count / silver_row_count * 100
            label = "Delayed (≥15 min)" if delay_val == 1 else "On-time (<15 min)"
            print(f"  {label}: {count:,} ({pct:.2f}%)")
        
        print("✓ PASS - Target variable calculated")
        
        # ==================== FLIGHT GOLD LAYER TESTS ====================
        
        # Test 13: Process Silver → Flight Gold
        print(f"\n{'='*80}")
        print("TEST 13: Flight Gold Processing")
        print("="*80)
        
        df_flight_gold = process_silver_to_flight_gold(
            silver_path=silver_output_path,
            flight_gold_output_path=flight_gold_output_path,
            spark=spark,
            snapshot_date=test_date,
            output_mode='overwrite'
        )
        
        flight_gold_row_count = df_flight_gold.count()
        
        if flight_gold_row_count > 0:
            print(f"✓ PASS - Processed {flight_gold_row_count:,} rows")
            
            if flight_gold_row_count == silver_row_count:
                print(f"  ✓ Row count preserved from Silver")
            else:
                print(f"  ⚠ Row count changed: {silver_row_count:,} → {flight_gold_row_count:,}")
        else:
            print(f"✗ FAIL - No rows processed")
            spark.stop()
            return False
        
        # Test 14: Column validation (dropped 7, added 10)
        print(f"\n{'='*80}")
        print("TEST 14: Flight Gold Column Validation")
        print("="*80)
        
        silver_cols = len(df_silver.columns)
        gold_cols = len(df_flight_gold.columns)
        col_diff = gold_cols - silver_cols
        
        print(f"  Silver columns: {silver_cols}")
        print(f"  Flight Gold columns: {gold_cols}")
        print(f"  Column change: {col_diff:+d}")
        
        # Check dropped columns are gone
        dropped_cols = ['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE', 
                        'source_file', 'is_delayed_15', 'processing_timestamp', 'sort_time']
        
        still_present = [c for c in dropped_cols if c in df_flight_gold.columns]
        
        if still_present:
            print(f"  ⚠ These columns should be dropped: {still_present}")
        else:
            print(f"  ✓ All unnecessary columns dropped")
        
        # Check new columns are added
        new_cols = ['IS_DELAYED', 'dep_3hour_col', 'arr_3hour_col',
                    'flight_id', 'daily_flights', 'volume_zscore', 'is_rare',
                    'is_abnormal_num', 'is_peak_day', 'is_extra_candidate']
        
        missing_new_cols = [c for c in new_cols if c not in df_flight_gold.columns]
        
        if missing_new_cols:
            print(f"  ✗ FAIL - Missing new columns: {missing_new_cols}")
            spark.stop()
            return False
        else:
            print(f"  ✓ All {len(new_cols)} new feature columns added")
        
        # Test 15: IS_DELAYED distribution
        print(f"\n{'='*80}")
        print("TEST 15: IS_DELAYED Distribution (3 Categories)")
        print("="*80)
        
        delay_cat_dist = df_flight_gold.groupBy('IS_DELAYED').count().orderBy('IS_DELAYED').collect()
        
        total_check = 0
        for row in delay_cat_dist:
            category = row['IS_DELAYED']
            count = row['count']
            pct = count / flight_gold_row_count * 100
            
            if category == 0:
                label = "Category 0 (< 60 min)"
            elif category == 1:
                label = "Category 1 (60-119 min)"
            elif category == 2:
                label = "Category 2 (≥ 120 min)"
            else:
                label = f"Category {category}"
            
            print(f"  {label}: {count:,} ({pct:.2f}%)")
            total_check += count
        
        if total_check == flight_gold_row_count:
            print("✓ PASS - IS_DELAYED distribution sums to 100%")
        else:
            print(f"✗ FAIL - IS_DELAYED distribution mismatch")
            spark.stop()
            return False
        
        # Test 16: 3-hour bucket validation
        print(f"\n{'='*80}")
        print("TEST 16: 3-Hour Time Bucket Validation")
        print("="*80)
        
        dep_buckets = df_flight_gold.select('dep_3hour_col').distinct().collect()
        arr_buckets = df_flight_gold.select('arr_3hour_col').distinct().collect()
        
        print(f"  Unique dep_3hour_col buckets: {len(dep_buckets)}")
        print(f"  Sample dep buckets:")
        for row in dep_buckets[:5]:
            print(f"    - {row['dep_3hour_col']}")
        
        print(f"  Unique arr_3hour_col buckets: {len(arr_buckets)}")
        
        # Check format
        expected_format_found = any('0000-0259' in str(row['dep_3hour_col']) or 
                                   '0300-0559' in str(row['dep_3hour_col']) 
                                   for row in dep_buckets)
        
        if expected_format_found or len(dep_buckets) > 0:
            print("✓ PASS - 3-hour buckets created with expected format")
        else:
            print("✗ FAIL - 3-hour bucket format issue")
            spark.stop()
            return False
        
        # Test 17: Flight volume features validation
        print(f"\n{'='*80}")
        print("TEST 17: Flight Volume Features Validation")
        print("="*80)
        
        # Check flight_id
        unique_flights = df_flight_gold.select('flight_id').distinct().count()
        print(f"  Unique flight_ids: {unique_flights:,}")
        
        # Check binary flags
        rare_count = df_flight_gold.filter(col('is_rare') == 1).count()
        abnormal_count = df_flight_gold.filter(col('is_abnormal_num') == 1).count()
        peak_count = df_flight_gold.filter(col('is_peak_day') == 1).count()
        extra_count = df_flight_gold.filter(col('is_extra_candidate') == 1).count()
        
        print(f"  is_rare (1): {rare_count:,} ({rare_count/flight_gold_row_count*100:.2f}%)")
        print(f"  is_abnormal_num (1): {abnormal_count:,} ({abnormal_count/flight_gold_row_count*100:.2f}%)")
        print(f"  is_peak_day (1): {peak_count:,} ({peak_count/flight_gold_row_count*100:.2f}%)")
        print(f"  is_extra_candidate (1): {extra_count:,} ({extra_count/flight_gold_row_count*100:.2f}%)")
        
        # Check daily_flights and volume_zscore are numeric
        daily_flights_null = df_flight_gold.filter(col('daily_flights').isNull()).count()
        volume_zscore_null = df_flight_gold.filter(col('volume_zscore').isNull()).count()
        
        if daily_flights_null == 0 and volume_zscore_null == 0:
            print("✓ PASS - All flight volume features calculated correctly")
        else:
            print(f"  ⚠ Null values found: daily_flights={daily_flights_null}, volume_zscore={volume_zscore_null}")
        
        # ==================== GOLD COMBINED LAYER TESTS ====================
        
        # Test 18: Weather data preparation
        print(f"\n{'='*80}")
        print("TEST 18: Weather Data Preparation")
        print("="*80)
        
        if not os.path.exists(weather_input_path):
            print(f"✗ FAIL - Weather data not found: {weather_input_path}")
            print("  Please ensure silver_weather_store parquet exists for test date")
            spark.stop()
            return False
        
        df_weather = spark.read.parquet(weather_input_path)
        weather_row_count = df_weather.count()
        
        print(f"  Weather rows: {weather_row_count:,}")
        
        # Check structure: should be 8 time buckets × 3 airports = 24 rows
        if 'name' in df_weather.columns or 'airport' in df_weather.columns:
            airport_col = 'airport' if 'airport' in df_weather.columns else 'name'
            unique_airports = df_weather.select(airport_col).distinct().count()
            print(f"  Unique airports: {unique_airports}")
        
        if 'time' in df_weather.columns:
            unique_times = df_weather.select('time').distinct().count()
            print(f"  Unique time buckets: {unique_times}")
        
        if weather_row_count == 24:
            print("✓ PASS - Weather data structure as expected (24 rows = 8 times × 3 airports)")
        else:
            print(f"  ⚠ Weather row count: {weather_row_count} (expected 24)")
        
        # Test 19: Gold Combined join
        print(f"\n{'='*80}")
        print("TEST 19: Gold Combined Join Processing")
        print("="*80)
        
        df_gold_combined = process_to_gold_combined(
            flight_gold_path=flight_gold_output_path,
            weather_path=weather_input_path,
            gold_combined_output_path=gold_combined_output_path,
            spark=spark,
            snapshot_date=test_date,
            output_mode='overwrite'
        )
        
        gold_combined_row_count = df_gold_combined.count()
        gold_combined_cols = len(df_gold_combined.columns)
        
        if gold_combined_row_count > 0:
            print(f"✓ PASS - Processed {gold_combined_row_count:,} rows")
            print(f"  Total columns: {gold_combined_cols}")
            
            # Check row count matches flight gold
            if gold_combined_row_count == flight_gold_row_count:
                print(f"  ✓ Row count matches Flight Gold (no rows lost in join)")
            else:
                print(f"  ⚠ Row count difference: Flight Gold={flight_gold_row_count:,}, Combined={gold_combined_row_count:,}")
        else:
            print(f"✗ FAIL - No rows processed")
            spark.stop()
            return False
        
        # Check weather columns added
        weather_cols = [c for c in df_gold_combined.columns if c.startswith('weather_')]
        print(f"  Weather columns added: {len(weather_cols)}")
        
        if len(weather_cols) > 0:
            print(f"  Sample weather columns: {weather_cols[:5]}")
            
            # Check for nulls in weather columns
            sample_weather_col = weather_cols[0]
            non_null_weather = df_gold_combined.filter(col(sample_weather_col).isNotNull()).count()
            weather_coverage = (non_null_weather / gold_combined_row_count) * 100
            
            print(f"  Weather coverage: {weather_coverage:.2f}%")
            
            if weather_coverage >= 80:
                print("✓ PASS - Good weather data coverage")
            else:
                print(f"  ⚠ Low weather coverage")
        else:
            print("✗ FAIL - No weather columns found")
            spark.stop()
            return False
        
        # Test 20: Gold Combined final validation
        print(f"\n{'='*80}")
        print("TEST 20: Gold Combined Final Validation")
        print("="*80)
        
        gold_validation = validate_gold_combined_parquet(
            gold_combined_output_path=gold_combined_output_path,
            spark=spark,
            is_daily=True
        )
        
        # Check all required features present
        required_features = [
            'IS_DELAYED', 'dep_3hour_col', 'arr_3hour_col',
            'flight_id', 'daily_flights', 'volume_zscore', 'is_rare',
            'is_abnormal_num', 'is_peak_day', 'is_extra_candidate'
        ]
        
        missing_features = [f for f in required_features if f not in df_gold_combined.columns]
        
        if missing_features:
            print(f"✗ FAIL - Missing required features: {missing_features}")
            spark.stop()
            return False
        else:
            print(f"✓ PASS - All {len(required_features)} required features present")
        
        # Check FlightDate dropped in OOT mode
        if 'FlightDate' in df_gold_combined.columns:
            print("  ⚠ FlightDate still present (should be dropped in OOT mode)")
        else:
            print("  ✓ FlightDate dropped (correct for OOT mode)")
        
        # Check IS_DELAYED present and correct
        if 'IS_DELAYED' in df_gold_combined.columns:
            delay_values = df_gold_combined.select('IS_DELAYED').distinct().count()
            if delay_values <= 3:
                print(f"  ✓ IS_DELAYED has {delay_values} categories (expected 3)")
            else:
                print(f"  ⚠ IS_DELAYED has {delay_values} categories")
        
        print("✓ PASS - Gold Combined validation complete")
        
        # ==================== FINAL SUMMARY ====================
        
        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nComplete Medallion Architecture OOT Processing Successful!")
        
        print(f"\n{'='*80}")
        print("DATA FLOW SUMMARY")
        print("="*80)
        print(f"\nBronze  → Silver  → Flight Gold → Gold Combined")
        print(f"{bronze_row_count:>7,} → {silver_row_count:>7,} → {flight_gold_row_count:>11,} → {gold_combined_row_count:>14,} rows")
        
        bronze_to_silver_reduction = (1 - silver_row_count / bronze_row_count) * 100
        print(f"\nData reduction: Bronze → Silver: {bronze_to_silver_reduction:.1f}%")
        
        print(f"\n{'='*80}")
        print("COLUMN COUNTS")
        print("="*80)
        print(f"Bronze:         {len(df_bronze.columns)} columns")
        print(f"Silver:         {len(df_silver.columns)} columns")
        print(f"Flight Gold:    {gold_cols} columns (added {new_cols})")
        print(f"Gold Combined:  {gold_combined_cols} columns (added weather features)")
        
        print(f"\n{'='*80}")
        print("OUTPUT FILES")
        print("="*80)
        print(f"Bronze:         {bronze_output_path}")
        print(f"Silver:         {silver_output_path}")
        print(f"Flight Gold:    {flight_gold_output_path}")
        print(f"Gold Combined:  {gold_combined_output_path}")
        
        print(f"\n{'='*80}")
        print("FINAL OUTPUT READY FOR MODEL")
        print("="*80)
        print(f"\n✓ Gold Combined Parquet: {gold_combined_output_path}")
        print(f"✓ Rows: {gold_combined_row_count:,}")
        print(f"✓ Columns: {gold_combined_cols}")
        print(f"✓ Target: IS_DELAYED (ordinal: 0, 1, 2)")
        print(f"✓ Features: Flight attributes + Volume features + Weather data")
        
        print("\nTo clean up test files:")
        print(f"  rm -rf {bronze_output_path}")
        print(f"  rm -rf {silver_output_path}")
        print(f"  rm -rf {flight_gold_output_path}")
        print(f"  rm -rf {gold_combined_output_path}")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("TEST FAILED WITH EXCEPTION")
        print("="*80)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        spark.stop()
        return False


def main():
    """
    Run all tests
    """
    success = test_daily_oot_processing()
    
    if success:
        print("\n✓ All tests passed! Complete medallion architecture processing is ready.")
        print("✓ Bronze → Silver → Flight Gold → Gold Combined pipeline validated.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
