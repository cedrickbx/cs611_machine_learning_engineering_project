# Medallion Architecture - Gold Layer Implementation

## Overview
Complete implementation of Silver → Gold layer processing for Flight Delay Prediction using Medallion Architecture.

---

## Files Delivered

### Phase 1: Silver Flight Store
**File**: `silver_flight_store.py`
- Execution script for Bronze → Silver processing
- Mirrors structure of `bronze_flight_store.py`
- Supports dual mode: Historical batch and Daily OOT
- CLI: `--snapshotdate` parameter
- Output: `silver_flight_oot_{date}.parquet` or `silver_flight_historical.parquet`

### Phase 2: Flight Gold Processing

#### Step 1-4: Processing Functions
**File**: `data_processing_flight_gold.py`
- Feature engineering functions (no execution)
- Functions implemented:
  1. `drop_unnecessary_columns()` - Remove 7 unnecessary columns
  2. `create_delay_categories()` - Create IS_DELAYED (ordinal: 0, 1, 2)
  3. `create_3hour_buckets()` - Create dep_3hour_col and arr_3hour_col
  4. `create_flight_volume_features()` - Create 7 volume/scheduling features

#### Step 5: Flight Gold Execution
**File**: `gold_flight_store.py`
- Execution script for Silver → Flight Gold
- Imports from `data_processing_flight_gold.py`
- Orchestrates feature engineering pipeline
- Output: `flight_gold_oot_{date}.parquet` or `flight_gold_historical.parquet`

### Phase 2B: Gold Combined Processing
**File**: `gold_combined_store.py`
- Self-contained execution script
- Joins Flight Gold + Weather data
- All functions included in same file
- Join logic:
  - NYC departures: Join on (date, ORIGIN, time_bucket)
  - NYC arrivals: Join on (date, DEST, time_bucket)
  - Weather data broadcast to all flights in 3-hour bucket
- OOT mode: Drops FlightDate column
- Output: `gold_combined_oot_{date}.parquet` or `gold_combined_historical.parquet`

### Testing
**File**: `test_daily_oot.py` (Extended)
- Complete end-to-end test: Bronze → Silver → Gold
- 20 tests covering:
  - Tests 1-7: Bronze layer validation
  - Tests 8-12: Silver layer validation
  - Tests 13-17: Flight Gold validation
  - Tests 18-20: Gold Combined validation
- Validates complete medallion architecture

---

## Feature Engineering Summary

### Dropped Columns (7):
- CANCELLED, DIVERTED, CANCELLATION_CODE
- source_file, is_delayed_15, processing_timestamp, sort_time

### New Features Added (10):

#### 1. Target Variable (1 feature)
- **IS_DELAYED**: Ordinal delay categories
  - 0: < 60 minutes
  - 1: 60-119 minutes
  - 2: ≥ 120 minutes

#### 2. Time Buckets (2 features)
- **dep_3hour_col**: Departure 3-hour bucket (e.g., "0600-0859")
- **arr_3hour_col**: Arrival 3-hour bucket (e.g., "0600-0859")

#### 3. Flight Volume Features (7 features)
- **flight_id**: Carrier + Flight number identifier
- **daily_flights**: Count per carrier-origin-date
- **volume_zscore**: Z-score of daily flights
- **is_rare**: Flag if flight appears <3 times in dataset
- **is_abnormal_num**: Flag if flight number >8000
- **is_peak_day**: Flag if volume z-score >2
- **is_extra_candidate**: Composite flag for special/extra flights

#### 4. Weather Features (14+ features)
- Joined from `silver_weather_store.parquet`
- Includes: wind_dir_deg, wind_speed_mps, temp_c, dewpoint_c, visibility_m, ceiling_m, slp_hpa, etc.
- Matched to flights by 3-hour time buckets and airport
- Same weather data replicated for all flights in bucket

---

## Data Flow

```
Bronze (NYC flights)
    ↓
Silver (cleaned, validated)
    ↓
Flight Gold (features engineered)
    ↓ + Weather (3-hour buckets)
Gold Combined (final dataset for XGBoost)
```

### Column Evolution
- **Bronze**: ~40 columns
- **Silver**: ~40 columns (cleaned)
- **Flight Gold**: ~43 columns (dropped 7, added 10)
- **Gold Combined**: ~57 columns (added 14 weather features)

---

## Usage Examples

### Daily OOT Processing (Inference)
```bash
# Process single day through entire pipeline
python silver_flight_store.py --snapshotdate 2025-01-15
python gold_flight_store.py --snapshotdate 2025-01-15
python gold_combined_store.py --snapshotdate 2025-01-15

# Test entire pipeline
python test_daily_oot.py
```

### Historical Batch Processing (Training)
```bash
# Process all 24 months for model training
python silver_flight_store.py
python gold_flight_store.py
python gold_combined_store.py
```

---

## Output Files

### OOT Mode (Daily Inference)
```
datamart/silver/flight/silver_flight_oot_2025_01_15.parquet/
datamart/gold/flight/flight_gold_oot_2025_01_15.parquet/
datamart/gold/combined/gold_combined_oot_2025_01_15.parquet/
```
- **Note**: FlightDate column is DROPPED in Gold Combined (OOT mode)

### Historical Mode (Training)
```
datamart/silver/flight/silver_flight_historical.parquet/
datamart/gold/flight/flight_gold_historical.parquet/
datamart/gold/combined/gold_combined_historical.parquet/
```
- **Note**: FlightDate column is KEPT in Gold Combined (Historical mode)

---

## Key Design Decisions

### 1. Weather Join Strategy
- **Departures**: Match ORIGIN airport to weather
- **Arrivals**: Match DEST airport to weather
- **NYC airports**: JFK, LGA, EWR (weather differences insignificant)
- **Time matching**: 3-hour buckets align with weather time intervals
- **Broadcast**: Same weather data for all flights in bucket

### 2. Time Bucket Mapping
```
Flight 3-hour bucket → Weather time bucket
"0000-0259" → "0000"
"0300-0559" → "0300"
"0600-0859" → "0600"
... and so on
```

### 3. OOT vs Historical Differences
- **OOT**: Single date, FlightDate dropped in final output
- **Historical**: Multiple dates (730 days), FlightDate kept for joining
- **Partitioning**: Both modes partition by snapshot_date

### 4. Target Variable Design
- Changed from binary (is_delayed_15) to ordinal (IS_DELAYED)
- 3 categories allow for multi-class classification
- Based on arrival/departure delay (primary passenger experience metric)

---

## Testing & Validation

### Test Coverage (20 tests)
1. CSV file discovery
2. Bronze processing
3. Bronze validation
4. Bronze partitioning
5. Bronze derived columns
6. Single date constraint
7. NYC filter verification
8. Silver processing
9. Silver validation
10. Silver partitioning
11. Data cleaning validation
12. Target variable distribution
13. **Flight Gold processing**
14. **Column validation (drops/adds)**
15. **IS_DELAYED distribution**
16. **3-hour bucket validation**
17. **Volume features validation**
18. **Weather data preparation**
19. **Gold Combined join**
20. **Final validation**

### Success Criteria
✅ All 20 tests pass
✅ `gold_combined_oot_{date}.parquet` created
✅ Flight features + weather features present
✅ IS_DELAYED ordinal target (0, 1, 2)
✅ >80% weather coverage
✅ Ready for XGBoost training/inference

---

## Next Steps

1. **Run test suite**: `python test_daily_oot.py`
2. **Process historical data**: Scale to 730 days for training set
3. **Train XGBoost model**: Use `gold_combined_historical.parquet`
4. **Daily inference**: Process daily OOT data for predictions
5. **Model evaluation**: Compare predictions vs actual IS_DELAYED labels

---

## File Structure Summary

```
Project Root/
├── silver_flight_store.py              # Phase 1
├── data_processing_flight_gold.py      # Phase 2 (Steps 1-4)
├── gold_flight_store.py                # Phase 2 (Step 5)
├── gold_combined_store.py              # Phase 2B
├── test_daily_oot.py                   # Extended testing
└── datamart/
    ├── bronze/flight/
    ├── silver/flight/
    ├── silver/weather/
    └── gold/
        ├── flight/
        └── combined/                   # FINAL OUTPUT
```

---

## Dependencies
- PySpark
- Python 3.x
- utilities:
  - `utils/data_processing_flight_bronze.py`
  - `utils/data_processing_flight_silver.py`
  - `utils/data_processing_flight_gold.py`

---

**Status**: ✅ Complete and ready for deployment on January 1, 2025
Run inference for period January 1, 2025 to February 28, 2025
Retraining on February 28, 2025
Deployment with retrained model on March 1, 2025

**Date**: Nov 1, 2025
