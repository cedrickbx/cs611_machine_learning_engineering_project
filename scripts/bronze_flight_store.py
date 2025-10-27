"""
Main Script: Bronze Layer Processing - MODIFIED FOR OOT
Process flight data to Bronze layer with support for:
1. Historical batch processing (24 months)
2. Daily OOT processing (single day)

Usage:
    # Historical batch (default)
    python main_static.py
    
    # Daily OOT processing
    python main_static.py --snapshotdate 2025-01-01

Output:
    Historical: datamart/bronze/flight/bronze_flight_historical.parquet/
    Daily OOT:  datamart/bronze/flight/bronze_flight_oot.parquet/
"""

import os
import time
import argparse
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
import sys
from pathlib import Path

# Import our processing functions
from utils.data_processing_flight_bronze import (
    process_all_months_to_bronze,
    validate_bronze_parquet,
    print_holiday_list
)


def main(snapshotdate=None):
    """
    Main execution function
    
    Args:
        snapshotdate: Optional date string "YYYY-MM-DD" for daily OOT processing
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FLIGHT DELAY PREDICTION - BRONZE LAYER PROCESSING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine processing mode
    if snapshotdate:
        print(f"Mode: DAILY OOT")
        print(f"Snapshot Date: {snapshotdate}")
    else:
        print(f"Mode: HISTORICAL BATCH")
    
    print("="*80 + "\n")
    
    # Configuration - different paths for historical vs OOT
    if snapshotdate:
        data_directory = "../data/flight/oot/" # OOT CSV files 
        bronze_output_path = "../datamart/bronze/flight/bronze_flight_oot.parquet" 
        is_daily = True
    else:
        data_directory = "../data/flight/train/" # Historical CSV files 
        bronze_output_path = "../datamart/bronze/flight/bronze_flight_historical.parquet" 
        is_daily = False
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory not found: {data_directory}")
        print("\nExpected structure:")
        if snapshotdate:
            print(f"  {data_directory}T_ONTIME_REPORTING-01_25.csv")
            print(f"  {data_directory}T_ONTIME_REPORTING-02_25.csv")
            print("  ... (OOT month CSVs)")
        else:
            print(f"  {data_directory}T_ONTIME_REPORTING-01_23.csv")
            print(f"  {data_directory}T_ONTIME_REPORTING-02_23.csv")
            print("  ...")
            print(f"  {data_directory}T_ONTIME_REPORTING-12_24.csv")
        sys.exit(1)  # non-zero -> task failure in Airflow
        return

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelayBronze") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to reduce noise
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")

    # Print US federal holidays for review (only in historical mode)
    if not snapshotdate:
        print_holiday_list()
    
    try:
        # Process to Bronze
        df_bronze = process_all_months_to_bronze(
            data_directory=data_directory,
            bronze_output_path=bronze_output_path,
            spark=spark,
            snapshot_date=snapshotdate,
            output_mode='overwrite'
        )
        
        # Validate outputs
        validation_results = validate_bronze_parquet(
            bronze_output_path=bronze_output_path,
            spark=spark,
            is_daily=is_daily
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Bronze Parquet created: {bronze_output_path}")
        print(f"✓ Total rows: {validation_results.get('total_rows', 'N/A'):,}")
        print(f"✓ Date range: {validation_results.get('min_date', 'N/A')} to {validation_results.get('max_date', 'N/A')}")
        print(f"✓ Processing time: {elapsed_time/60:.1f} minutes")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if snapshotdate:
            # Daily OOT mode
            print(f"\n✓ Daily Bronze created for {snapshotdate}")
            print("\nNext:")
            print("1. Process to Silver layer (feature engineering)")
            print("2. Process to Gold layer (join with weather)")
            print("3. Run model inference")
            print("4. Compare prediction vs actual label")
            print("\nTo process next day:")
            print(f"  python main_static.py --snapshotdate YYYY-MM-DD")
        else:
            # Historical mode
            print("\n1. Review holiday list above to verify accuracy")
            print("2. Inspect Parquet file:")
            print(f"   ls -lh {bronze_output_path}/")
            print("\n3. Next: Build XGBoost model using this Bronze data")
            print("\n4. Later: Process OOT data daily with --snapshotdate flag")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print("="*80)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        print("\nStopping Spark session...")
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Bronze Layer Processing for Flight Delay Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 24 months (historical batch)
  python main_static.py
  
  # Process single day for OOT inference
  python main_static.py --snapshotdate 2025-01-01
  
  # Process another day
  python main_static.py --snapshotdate 2025-01-02
        """
    )
    
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=False,
        default=None,
        help="Snapshot date for daily OOT processing (format: YYYY-MM-DD). If omitted, processes all historical data."
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
    
    # Call main with arguments
    main(snapshotdate=args.snapshotdate)