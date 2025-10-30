from utils import data_processing_weather_silver
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

def main():
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create silver datalake
    bronze_weather = "../datamart/bronze/weather_history/"
    silver_weather = "../datamart/silver/weather_history/"

    if not os.path.exists(silver_weather):
        os.makedirs(silver_weather)
    # run data processing
    data_processing_weather_silver.process_main_weather_spark(spark, bronze_weather, silver_weather)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    main()
