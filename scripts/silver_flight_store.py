"""
Main Script: Silver Layer Processing
Process flight data from Bronze to Silver layer with support for:
1. Historical batch processing (24 months)
2. Daily OOT processing (single day)

Usage:
    # Historical batch (default)
    python silver_flight_store.py
    
    # Daily OOT processing
    python silver_flight_store.py --snapshotdate 2025-01-15

Output:
    Historical: datamart/silver/flight/silver_flight_historical.parquet/
    Daily OOT:  datamart/silver/flight/silver_flight_oot_YYYY_MM_DD.parquet/
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
from utils.data_processing_flight_silver import (
    process_bronze_to_silver,
    validate_silver_parquet
)


def main(snapshotdate=None):
    """
    Main execution function
    
    Args:
        snapshotdate: Optional date string "YYYY-MM-DD" for daily OOT processing
    """
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
    
    # ------------------------------------------------------------------
    # Paths (separate output; OOT includes the date in the filename)
    # ------------------------------------------------------------------
    if snapshotdate:
        # OOT mode - read from Bronze OOT
        snap_tag = snapshotdate.replace("-", "_")
        bronze_path = f"datamart/bronze/flight/bronze_flight_oot_{snap_tag}.parquet"
        silver_output_path = f"datamart/silver/flight/silver_flight_oot_{snap_tag}.parquet"
        is_daily = True
    else:
        # Historical mode
        bronze_path = "datamart/bronze/flight/bronze_flight_historical.parquet"
        silver_output_path = "datamart/silver/flight/silver_flight_historical.parquet"
        is_daily = False
    
    # Check if bronze directory exists
    if not os.path.exists(bronze_path):
        print(f"ERROR: Bronze data not found: {bronze_path}")
        print("\nPlease run Bronze processing first:")
        if snapshotdate:
            print(f"  python bronze_flight_store.py --snapshotdate {snapshotdate}")
        else:
            print(f"  python bronze_flight_store.py")
        sys.exit(1)
        return

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FlightDelaySilver") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to reduce noise
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session initialized\n")

    try:
        # Process to Silver
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
            # Daily OOT mode
            print(f"\n✓ Daily Silver created for {snapshotdate}")
            print("\nNext:")
            print("1. Process to Flight Gold layer (feature engineering)")
            print(f"   python gold_flight_store.py --snapshotdate {snapshotdate}")
            print("2. Join with weather to create Gold Combined")
            print(f"   python gold_combined_store.py --snapshotdate {snapshotdate}")
        else:
            # Historical mode
            print("\n1. Inspect Silver Parquet file:")
            print(f"   ls -lh {silver_output_path}/")
            print("\n2. Next: Process to Gold layer (feature engineering)")
            print("   python gold_flight_store.py")
        
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
        description="Silver Layer Processing for Flight Delay Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 24 months (historical batch)
  python silver_flight_store.py
  
  # Process single day for OOT inference
  python silver_flight_store.py --snapshotdate 2025-01-15
  
  # Process another day
  python silver_flight_store.py --snapshotdate 2025-01-16
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
            print("Expected format: YYYY-MM-DD (e.g., 2025-01-15)")
            exit(1)
    
    # Call main with arguments
    main(snapshotdate=args.snapshotdate)
