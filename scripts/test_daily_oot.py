"""
Test Script: Daily OOT Bronze → Silver Processing

Tests the daily OOT processing functionality by:
1. Processing a sample date through Bronze layer
2. Processing Bronze → Silver layer
3. Validating both outputs
4. Comparing with expected behavior

Usage:
    python test_daily_oot.py
"""

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession

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


def test_daily_oot_processing():
    """
    Test daily OOT Bronze → Silver processing
    """
    print("="*80)
    print("TEST: DAILY OOT BRONZE → SILVER PROCESSING")
    print("="*80)
    
    # Test configuration
    test_date = "2025-01-15"  # Mid-month to ensure CSV exists
    data_directory = "data/flight/oot/"
    bronze_output_path = "datamart/bronze/flight/test_bronze_oot.parquet"
    silver_output_path = "datamart/silver/flight/test_silver_oot.parquet"
    
    print(f"\nTest Date: {test_date}")
    print(f"Data Directory: {data_directory}")
    print(f"Bronze Output: {bronze_output_path}")
    print(f"Silver Output: {silver_output_path}")
    
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
        
        from pyspark.sql.functions import col
        
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
        
        # # Test 12: Verify Silver data quality
        # print(f"\n{'='*80}")
        # print("TEST 12: Silver Data Quality Checks")
        # print("="*80)
        
        # # Check for outliers
        # long_flights = df_silver.filter(col('ACTUAL_ELAPSED_TIME') > 1440).count()
        # long_distance = df_silver.filter(col('DISTANCE') > 6000).count()
        
        # print(f"  Flights >24 hours: {long_flights}")
        # print(f"  Flights >6000 miles: {long_distance}")
        
        # if long_flights == 0 and long_distance == 0:
        #     print("✓ PASS - No outliers in Silver data")
        # else:
        #     print("⚠ WARNING - Found outliers in Silver data")
        
        # Test 13: Verify target variable distribution
        print(f"\n{'='*80}")
        print("TEST 13: Target Variable Distribution")
        print("="*80)
        
        delay_dist = df_silver.groupBy('is_delayed_15').count().collect()
        
        for row in delay_dist:
            delay_val = row['is_delayed_15']
            count = row['count']
            pct = count / silver_row_count * 100
            label = "Delayed (≥15 min)" if delay_val == 1 else "On-time (<15 min)"
            print(f"  {label}: {count:,} ({pct:.2f}%)")
        
        print("✓ PASS - Target variable calculated")
        
        # All tests passed
        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nDaily OOT Bronze → Silver processing is working correctly!")
        print(f"\nBronze output: {bronze_output_path}")
        print(f"Silver output: {silver_output_path}")
        print(f"\nData flow: {bronze_row_count:,} rows → {silver_row_count:,} rows ({reduction_pct:.1f}% reduction)")
        print("\nTo clean up test files:")
        print(f"  rm -rf {bronze_output_path}")
        print(f"  rm -rf {silver_output_path}")
        
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
        print("\n✓ All tests passed! Daily OOT Bronze → Silver processing is ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()





