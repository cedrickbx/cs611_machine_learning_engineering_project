"""
Test Script: Daily OOT Bronze Processing

Tests the daily OOT processing functionality by:
1. Processing a sample date
2. Validating output
3. Comparing with expected behavior

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


def test_daily_oot_processing():
    """
    Test daily OOT Bronze processing
    """
    print("="*80)
    print("TEST: DAILY OOT BRONZE PROCESSING")
    print("="*80)
    
    # Test configuration
    test_date = "2025-01-15"  # Mid-month to ensure CSV exists
    data_directory = "data/flight/oot/"
    bronze_output_path = "datamart/bronze/flight/test_bronze_oot.parquet"
    
    print(f"\nTest Date: {test_date}")
    print(f"Data Directory: {data_directory}")
    print(f"Output Path: {bronze_output_path}")
    
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
        
        df = process_all_months_to_bronze(
            data_directory=data_directory,
            bronze_output_path=bronze_output_path,
            spark=spark,
            snapshot_date=test_date,
            output_mode='overwrite'
        )
        
        row_count = df.count()
        
        if row_count > 0:
            print(f"✓ PASS - Processed {row_count:,} rows")
        else:
            print(f"✗ FAIL - No rows processed")
            spark.stop()
            return False
        
        # Test 3: Validate output structure
        print(f"\n{'='*80}")
        print("TEST 3: Output Validation")
        print("="*80)
        
        validation_results = validate_bronze_parquet(
            bronze_output_path=bronze_output_path,
            spark=spark,
            is_daily=True
        )
        
        # Test 4: Check partitioning
        print(f"\n{'='*80}")
        print("TEST 4: Partition Structure")
        print("="*80)
        
        partition_dir = f"{bronze_output_path}/snapshot_date={test_date}"
        
        if os.path.exists(partition_dir):
            print(f"✓ PASS - Partition created: snapshot_date={test_date}")
            
            # List files in partition
            parquet_files = [f for f in os.listdir(partition_dir) if f.endswith('.parquet')]
            print(f"  Parquet files: {len(parquet_files)}")
            
            for f in parquet_files[:3]:  # Show first 3
                size_mb = os.path.getsize(os.path.join(partition_dir, f)) / 1024 / 1024
                print(f"  - {f} ({size_mb:.2f} MB)")
        else:
            print(f"✗ FAIL - Partition not found: {partition_dir}")
            spark.stop()
            return False
        
        # Test 5: Verify derived columns
        print(f"\n{'='*80}")
        print("TEST 5: Derived Columns")
        print("="*80)
        
        required_cols = [
            'FlightDate', 'year_month', 'sort_time', 'is_delayed_15',
            'DayOfWeek', 'IsWeekend', 'IsPublicHoliday', 'snapshot_date'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            print("✓ PASS - All required columns present")
            print(f"  Total columns: {len(df.columns)}")
        else:
            print(f"✗ FAIL - Missing columns: {missing_cols}")
            spark.stop()
            return False
        
        # Test 6: Check single date constraint
        print(f"\n{'='*80}")
        print("TEST 6: Single Date Constraint")
        print("="*80)
        
        unique_dates = df.select('FlightDate').distinct().count()
        
        if unique_dates == 1:
            print(f"✓ PASS - Single date constraint satisfied")
        else:
            print(f"✗ FAIL - Found {unique_dates} unique dates (expected 1)")
            spark.stop()
            return False
        
        # Test 7: NYC filter
        print(f"\n{'='*80}")
        print("TEST 7: NYC Metro Filter")
        print("="*80)
        
        from pyspark.sql.functions import col
        
        nyc_airports = ['JFK', 'LGA', 'EWR']
        nyc_count = df.filter(
            col('ORIGIN').isin(nyc_airports) | col('DEST').isin(nyc_airports)
        ).count()
        
        if nyc_count == row_count:
            print(f"✓ PASS - 100% NYC coverage ({nyc_count:,} rows)")
        else:
            print(f"✗ FAIL - Only {nyc_count/row_count*100:.1f}% NYC coverage")
            spark.stop()
            return False
        
        # All tests passed
        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nDaily OOT Bronze processing is working correctly!")
        print(f"\nTest output location: {bronze_output_path}")
        print("\nTo clean up test files:")
        print(f"  rm -rf {bronze_output_path}")
        
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
        print("\n✓ All tests passed! Daily OOT processing is ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()