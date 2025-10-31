from utils import data_processing_weather_bronze
import argparse
import os
import glob
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col

# to call this script: python silver_weather_store.py

def main(snapshotdate=None):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create silver datalake
    data_path = "../data/weather_history/"
    bronze_weather = "../datamart/bronze/weather_history/"

    if not os.path.exists(bronze_weather):
        os.makedirs(bronze_weather)
    # run data processing
    data_processing_weather_bronze.process_main_weather_spark(spark, data_path, bronze_weather, snapshotdate)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
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