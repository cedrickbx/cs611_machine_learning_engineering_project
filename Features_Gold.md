## Markdown file for Feature Engineering, Gold Layer

*** FEATURE Engineering ***

## DROP COLUMNS
1. Remove cancelled and diverted flights, cancellation_code [-3 features]

Functions: 

def remove_cancelled_diverted(df):
def remove_cancelled_with_delays(df):
def remove_cancellation_code (df):

2. Remove source_file, is_delayed_15, processing_timestamp, sort_time  [-4 features]

## NEW COLUMNS

1. Using DEP_DELAY_NEW, and ARR_DELAY_NEW, add column with ordinal y-labels for "IS_DELAYED": 
Category 0 - "is_delayed_less_than_59_mins", 
Caategory 1 - "is_delayed_60_to_119_mins",  
Category 2 - "is_delayed_more_than_120_mins" 

[+1 feature with 3 ordinal values, 0, 1 and 2. Move this column to the extreme right]

- Create the function from DEP_TIME for flights Departing 
- Create the function from ARR_TIME for flights Arriving 

- Print distribution by category of y-labels, base 100%

2. Add 3hour_col [+2 features]
dep_3hour_col : group flights into 3-hour bucket using DEP_TIME_BLK as follows 0000-0259, 0300-0559, and so on
arr_3hour_col : group flights into 3-hour bucket using ARR_TIME_BLK as follows 0000-0259, 0300-0559, and so on


3. Add Scheduled Flights and Extra Flights [+7 features]
（1）flight_id : Concatenate carrier code (OP_UNIQUE_CARRIER) and flight number (OP_CARRIER_FL_NUM) into a unique flight identifier string
（2）daily_flights : Count number of flights operated by each carrier (OP_UNIQUE_CARRIER) from each airport (ORIGIN) on each normalized flight date
（3）volume_zscore : Compute z-score of daily_flights relative to its long-term mean and standard deviation per carrier–origin pair
（4）is_rare : Count number of appearances per flight_id; flag as 1 if a flight occurs fewer than 3 times across the dataset
 (5) is_abnormal_num : Flag flight numbers numerically greater than 8000 as abnormal (these are often special or non-commercial flights)
（6）is_peak_day : Tag flights that occur on statistically high-traffic days (z-score greater than 2)
（7）is_extra_candidate : Flag a flight as an “extra flight candidate” if it meets any of the following: rare flight (is_rare == 1), abnormal flight number (is_abnormal_num == 1), rare flight occurring on a peak day, or abnormal flight number on a public holiday (ispublicholiday == 1).

*** GOLD PROCESSING *** 
## JOIN WEATHER GOLD WITH FLIGHT GOLD TO CREATE ONE PARQUET FILE

1) After processing data_processing_flight_gold.py, create a new gold_label_store.py
2) In gold_label_store, join the output from flight_gold and from silver_weather_store.parquet using a time series Index of FlightDate, then within each FlightDate, the CRS_DEP_TIME and CRS_ARR_TIME combined into FlightTime, sorted using ascending time within each FlightDate.  Do NOT use nested list for index, instead use two columns. 
3) For all flights departing or arriving within a 3hour_col, after matching to the 3-hour buckets in silver_weather_store.parquet, repeat the same weather data for each row within the same 3-hour bucket. 
4) Output the joined file to "gold_combined.parquet"
4) Join the data using a similar method for daily OOT data except there is no need for column FlightDate. All rows within each OOT file are for the same FlightDate.  

*** END OF FEATURE ENGINEERING AND GOLD PROCESSING ***