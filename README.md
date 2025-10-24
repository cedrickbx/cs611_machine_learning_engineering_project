
# CS611 Machine Learning Engineering — Group 3

End-to-end MLE project with **Airflow**-orchestrated pipelines, a reproducible **Docker** setup, and **JupyterLab** for EDA.

---

## Table of Contents
- [Repository Layout](#repository-layout)
- [Data Locations](#data-locations)
- [Quick Start](#quick-start)
- [Docker Compose Commands](#docker-compose-commands)
- [Service URLs](#service-urls)
- [Developing Pipelines](#developing-pipelines)
- [Conventions](#conventions)

---

## Repository Layout
```
.
├─ scripts/
│  ├─ data/                 # Raw input data (large files are NOT tracked by Git)
│  ├─ datamart/             # Pipeline outputs (features/labels/gold tables)
│  ├─ utils/                # Shared utility functions
│  ├─ bronze_label_store.py # Template task for Airflow (use as a reference)
│  └─ notebooks/            # EDA notebooks (Jupyter)
├─ Dockerfile
├─ docker-compose.yaml
├─ requirements.txt
└─ README.md
```

## Data Locations
- Place **raw data** in `scripts/data/`  
  *Bulky files are ignored by Git; store locally or in S3/GCS.*
- Pipeline **outputs (datamarts)** go in `scripts/datamart/`.

> Tip: Add `scripts/data/` and other large paths to `.gitignore` (already recommended).

---

## Quick Start
```bash
# (Linux/macOS only) set Airflow UID once per shell
export AIRFLOW_UID=$(id -u)

# Build images
docker compose build

# Start all services in the background
docker compose up -d

# Check everything is running
docker compose ps
```

## Service URLs
- **Airflow UI (Building ML Pipeline):** http://localhost:8080  
- **JupyterLab (EDA):** http://localhost:8892 # I have changed to port to avoid conflicting with other opened JupyterLab

  *(Token is printed in `docker compose logs -f jupyterlab`)*

---

## Developing Pipelines
- Start new Airflow tasks from the template: `scripts/bronze_label_store.py`.
- Put shared helpers in `scripts/utils/`.
- Write derived tables/features/labels into `scripts/datamart/`.
- Keep EDA notebooks in `scripts/notebooks/`.

---

## Docker Compose Commands

### Build
```bash
docker compose build
```

### Up (detached)
```bash
docker compose up -d
```

### Logs (follow all services)
```bash
docker compose logs -f
```

### List services
```bash
docker compose ps
```

### Restart one service (example: webserver)
```bash
docker compose restart airflow-webserver
```

### Down (stop containers, keep volumes/data)
```bash
docker compose down
```

### Down + remove volumes (⚠️ deletes persisted data)
```bash
docker compose down -v
```

---


---

## Conventions
- **Naming:** `snake_case`; one DAG/task per file where possible.
- **Dependencies:** add Python packages to `requirements.txt`.
- **Data policy:** do **not** commit large/raw data to Git.
- **Reproducibility:** prefer deterministic seeds/configs in tasks when feasible.

---
## NY Region Flight Delay Prediction - Medallion Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PySpark 3.5.5](https://img.shields.io/badge/pyspark-3.5.5-orange.svg)](https://spark.apache.org/)

---

## 🎯 Project Overview

**Objective**: Build a production-ready ML pipeline to predict flight delays (15+ minutes) in the New York City region using medallion architecture (Bronze → Silver → Gold).

**Scope**: Flights involving **JFK, LGA, and EWR airports only**

**Team Members**:
- **Flight Data Pipeline**: [Your Name] - Processing flight delay data
- **Weather Data Pipeline**: [Colleague Name] - Processing weather data
- **Integration**: Combined at Gold layer for ML modeling

**Timeline**:
- **Days 1-2**: ✅ Bronze Layer (Historical Data) - **CURRENT PHASE**
- **Day 3-4**: Silver/Gold Layers +
- **Days 5-6**:  Model Training (XGBoost + Ensemble)
- **Days 7-8**: Airflow DAG for OOT predictions 

---

## 📊 Data Sources

### **Flight Data**
- **Source**: US DOT On-Time Performance dataset
- **Format**: Monthly CSV files (31 columns per file)
- **Training Period**: Jan 2023 - Dec 2024 (24 months)
- **Expected Volume**: ~1M flights (after NYC filter)
- **Region Focus**: JFK, LGA, EWR airports only

### **Weather Data** 
(Handled by colleague in parallel pipeline)

---

## 🏗️ Architecture

```
Raw CSV Files (24 months)
    ↓
Bronze Layer (Single Parquet) ← YOU ARE HERE ON DAY 1
    ↓
Silver Layer (Feature Engineering) [Future]
    ↓
Gold Layer (ML-Ready Dataset) [Future]
    ↓
Model Training & Prediction [Future]
```

**Current Status**: ✅ **Bronze Layer Complete**
📦 Google Drive Storage: The raw CSV files zipped to 38 MB and Bronze Parquet file (~150 MB) are all stored on Google Drive for easy team access.
Link is here: https://drive.google.com/drive/folders/1atInyGuqLFnYhZke5Dvt_4wQqGTG8_0Q?usp=drive_link

---

## 📁 Repository Structure

```
cs611_machine_learning_engineering_project/
├── README.md                      # This file
├── .gitignore                     # Excludes data/ and datamart/
├── Dockerfile                     # PySpark + Jupyter environment
├── docker-compose.yaml            # Service orchestration
├── requirements.txt               # Python dependencies
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   └── data_processing_flight_bronze.py  # Bronze layer logic
│
├── main_static.py                 # Main Bronze processing script
├── test_setup.py                  # Environment validation
│
├── data/                          # NOT IN GIT - Local data
│   └── flight/
│       └── train/                # 24 monthly CSVs
│           ├── T_ONTIME_REPORTING-01_23.csv
│           ├── T_ONTIME_REPORTING-02_23.csv
│           └── ... (22 more files)
│
└── datamart/                      # NOT IN GIT - Pipeline outputs
    └── bronze/
        └── flight/
            └── bronze_flight_combined.parquet/  # ← Output
```

---

## 🚀 Quick Start

### **Prerequisites**
- Docker Desktop installed and running
- Git installed
- At least 10GB free disk space
- 24 CSV files downloaded from Google Drive/data

---

### **Step 1: Clone Repository**

```bash
git clone https://github.com/cedrickbx/cs611_machine_learning_engineering_project.git
cd cs611_machine_learning_engineering_project
```

---

### **Step 2: Add Data Files**

⚠️ **CRITICAL**: Data files are **NOT in GitHub** due to size. Each team member must download from Google Drive and place files locally.

Create this structure:
```
data/
└── flight/
    └── train/
        ├── T_ONTIME_REPORTING-01_23.csv
        ├── T_ONTIME_REPORTING-02_23.csv
        ├── T_ONTIME_REPORTING-03_23.csv
        ├── ... (21 more files)
        └── T_ONTIME_REPORTING-12_24.csv
```

**File naming must be exact**: `T_ONTIME_REPORTING-MM_YY.csv`

---

### **Step 3: Build Docker Environment** about 660 seconds

```bash
# Build Docker image
docker-compose build

# Start container
docker-compose up -d

# Verify container is running
docker ps
```

You should see `flight_delay_jupyter` container running.

---

### **Step 4: Access Container**

```bash
# Access container shell
docker exec -it flight_delay_jupyter bash

# You should now be in /app directory
```
(base) hclee@hclees-MacBook-Pro Group 3 Project % docker exec -it flight_delay_jupyter bash
(base) root@25452feb8114:/app# 
---

### **Step 5: Validate Environment**

```bash
# Inside container
python test_setup.py
```

**Expected Output**:
```
==========================================
ENVIRONMENT SETUP VALIDATION
==========================================

1. Testing Python version...
   ✓ PASS - Python 3.11+ detected

2. Testing package imports...
   ✓ pyspark
   ✓ pandas
   ✓ numpy
   ✓ holidays
   ✓ pyarrow
   ✓ PASS - All packages available

3. Testing Spark initialization...
   ✓ Spark session created
   ✓ Basic DataFrame operations working
   ✓ PASS - Spark initialization successful

4. Testing data directory...
   ✓ Data directory exists: data/flight/train/
   Found 24 CSV files
   ✓ PASS - All 24 CSV files present

5. Testing sample data loading...
   ✓ Loaded successfully
   ✓ NYC metro flights found
   ✓ PASS - Sample data loading successful

6. Testing output directory...
   ✓ Output directory ready
   ✓ PASS - Output directory accessible

==========================================
SUMMARY
==========================================
Tests Passed: 6/6

✓✓✓ ALL TESTS PASSED ✓✓✓

Environment is ready! You can now run:
  python main_static.py
```

If any tests fail, review error messages and fix before proceeding.

---

### **Step 6: Run Bronze Layer Processing**

```bash
# Inside container
python main_static.py
```

**Processing Steps**:
1. Loads 24 monthly CSV files (Jan 2023 - Dec 2024)
2. Filters for NYC metro airports (JFK, LGA, EWR)
3. Adds derived columns (target variable, temporal features, holidays)
4. Combines into single DataFrame
5. Sorts by FlightDate and departure/arrival time
6. Saves as Parquet (partitioned by year_month)
7. Runs validation checks

**Expected Runtime**: ~15-20 minutes (depending on system)

**Expected Output**:
```
================================================================================
BRONZE LAYER PROCESSING - FLIGHT DELAY DATA
================================================================================

Processing 24 months: Jan 2023 - Dec 2024

  Loading: T_ONTIME_REPORTING-01_23.csv
    Initial rows: 59,610
    After NYC filter: 49,339 (82.8%)

  Loading: T_ONTIME_REPORTING-02_23.csv
    Initial rows: 54,821
    After NYC filter: 45,392 (82.8%)

  ... (22 more months)

================================================================================
COMBINING DATA
================================================================================

  Combining all months...
  Total rows after union: 1,024,589

  Adding derived columns...
    ✓ year_month
    ✓ sort_time
    ✓ is_delayed_15
    ✓ DayOfWeek
    ✓ IsWeekend
    ✓ IsPublicHoliday
    ✓ processing_timestamp

  Sorting by FlightDate and sort_time...

================================================================================
SAVING TO PARQUET
================================================================================

  Output path: datamart/bronze/flight/bronze_flight_combined.parquet
  Partitioning by: year_month
  Compression: Snappy (default)

  Writing Parquet file...
  ✓ Parquet file saved successfully!

================================================================================
VALIDATION CHECKS
================================================================================

1. Total Rows: 1,024,589
   ✓ PASS - Row count within expected range

2. Date Range: 2023-01-01 to 2024-12-31
   ✓ PASS - Date range covers Jan 2023 to Dec 2024

3. Month Coverage: 24 unique months
   ✓ PASS - All 24 months present

4. NYC Metro Filter Validation:
   Flights FROM NYC metro: 590,234
   Flights TO NYC metro: 589,876
   Total NYC flights: 1,024,589
   ✓ PASS - 100% of flights involve NYC metro airports

5. Target Variable (is_delayed_15):
   On-time (<15 min): 791,234 (77.2%)
   Delayed (≥15 min): 233,355 (22.8%)

6. Cancelled Flights: 15,234 (1.5%)

7. Flights on Public Holidays: 35,678

================================================================================
PROCESSING COMPLETE
================================================================================

✓ Bronze Parquet created: datamart/bronze/flight/bronze_flight_combined.parquet
✓ Total rows: 1,024,589
✓ Date range: 2023-01-01 to 2024-12-31
✓ Processing time: 1.2 minutes

================================================================================
NEXT STEPS
================================================================================

1. Review holiday list above to verify accuracy
2. Inspect Parquet file:
   ls -lh datamart/bronze/flight/bronze_flight_combined.parquet/

3. Tomorrow: Build XGBoost model using this Bronze data

4. Later: Develop Silver/Gold layers and Airflow DAG
```

---

### **Step 7: Verify Output**

```bash
# Check Parquet structure
ls -lh datamart/bronze/flight/bronze_flight_combined.parquet/

# Expected structure:
# year_month=2023-01/
# year_month=2023-02/
# ... (24 partitions)
# year_month=2024-12/

exit ## Ctrl-D
```

### ** Step 8: Stop Docker
```bash
docker-compose down
** To Restart
```bash
docker-compose up -d
docker exec -it flight_delay_jupyter bash
docker ps
```

---

## 📋 Bronze Layer Specifications

### **Input Data**
- **24 monthly CSV files** (Jan 2023 - Dec 2024)
- **~50K-60K rows per file** before filtering
- **31 columns** from US DOT dataset

### **Processing Steps**

1. **Load CSVs**: Read each monthly file with PySpark
2. **Derive FlightDate**: Combine YEAR, MONTH, DAY_OF_MONTH
3. **Filter NYC Metro**: Keep only flights with `ORIGIN` or `DEST` in `['JFK', 'LGA', 'EWR']`
4. **Add Derived Columns**:
   - `year_month`: For Parquet partitioning
   - `sort_time`: CRS_DEP_TIME (if departing NYC) or CRS_ARR_TIME (if arriving NYC)
   - `is_delayed_15`: Target variable (1 if ARR_DELAY_NEW >= 15, else 0)
   - `DayOfWeek`: 1=Sunday, 2=Monday, ..., 7=Saturday
   - `IsWeekend`: 1 if Saturday/Sunday, else 0
   - `IsPublicHoliday`: 1 if US federal holiday (using `holidays` library), else 0
   - `source_file`: Original CSV filename
   - `processing_timestamp`: When Bronze record created
5. **Union All Months**: Combine 24 DataFrames into one
6. **Sort**: By FlightDate (ascending), then sort_time (ascending)
7. **Save Parquet**: Partitioned by year_month, Snappy compression

### **Output Schema** (36 columns)

| Column | Type | Description |
|--------|------|-------------|
| **FlightDate** | DATE | Primary temporal key (derived) |
| **year_month** | STRING | "YYYY-MM" for partitioning |
| **sort_time** | INT | Departure or arrival time for sorting |
| **is_delayed_15** | INT | Target: 1 if delayed ≥15 min |
| **DayOfWeek** | INT | 1=Sun...7=Sat |
| **IsWeekend** | INT | 1 if weekend |
| **IsPublicHoliday** | INT | 1 if federal holiday |
| YEAR | INT | Flight year |
| MONTH | INT | Flight month |
| DAY_OF_MONTH | INT | Day of month |
| OP_UNIQUE_CARRIER | STRING | Airline code |
| OP_CARRIER_FL_NUM | INT | Flight number |
| ORIGIN | STRING | Origin airport (JFK/LGA/EWR) |
| DEST | STRING | Destination airport |
| CRS_DEP_TIME | INT | Scheduled departure (HHMM) |
| DEP_TIME | FLOAT | Actual departure |
| DEP_DELAY_NEW | FLOAT | Departure delay (mins) |
| CRS_ARR_TIME | INT | Scheduled arrival (HHMM) |
| ARR_TIME | FLOAT | Actual arrival |
| ARR_DELAY_NEW | FLOAT | Arrival delay (mins) |
| CANCELLED | FLOAT | Cancelled indicator |
| DIVERTED | FLOAT | Diverted indicator |
| CARRIER_DELAY | FLOAT | Carrier delay cause |
| WEATHER_DELAY | FLOAT | Weather delay cause |
| NAS_DELAY | FLOAT | NAS delay cause |
| ... | ... | (Other original columns) |
| source_file | STRING | Source CSV name |
| processing_timestamp | TIMESTAMP | Processing time |

### **Output Volume**
- **Total rows**: ~1,000,000 flights
- **File size**: ~100-150 MB (compressed)
- **Partitions**: 24 (one per month)
- **Format**: Parquet with Snappy compression

---

## 🔍 Data Quality Validations

Automated checks included in `main_static.py`:

✅ **Row Count**: ~1M rows (900K-1.2M acceptable range)  
✅ **Date Coverage**: Jan 2023 - Dec 2024 (no gaps)  
✅ **Month Coverage**: All 24 months present  
✅ **NYC Filter**: 100% of flights involve JFK/LGA/EWR  
✅ **Target Distribution**: ~22-25% delayed flights  
✅ **Cancelled Flights**: ~1-2% (kept with NULL arrivals)  
✅ **Holiday Detection**: ~33 federal holidays marked  
✅ **Schema**: All derived columns present  

---

## 🛠️ Troubleshooting

### **Issue: Docker containers won't start**
```bash
# Check Docker status
docker ps -a

# Restart services
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs
```

### **Issue: CSV files not found**
- Verify files are in `data/flight/train/`
- Check filename format: `T_ONTIME_REPORTING-MM_YY.csv`
- Ensure files are CSV format (not .xlsx or compressed)

### **Issue: Spark initialization fails**
```bash
# Inside container, check Java
java -version

# Should show OpenJDK 11
```

### **Issue: Low row counts in output**
- Check if CSV files contain NYC airports (JFK/LGA/EWR)
- Review filtering logic in validation output
- Verify source data quality

### **Issue: Out of memory errors**
```bash
# Increase Docker memory allocation:
# Docker Desktop → Settings → Resources → Memory
# Recommended: 8GB+ for full processing
```
---

## 🔄 Daily OOT Processing (Out-of-Time Validation)

### **Overview**

The Bronze layer supports **two processing modes**:

1. **Historical Batch** (default): Process all 24 months for model training
2. **Daily OOT**: Process single day for out-of-time inference simulation

### **Use Cases**

| Mode | Use Case | Output | Command |
|------|----------|--------|---------|
| **Historical** | Initial model training | `bronze_flight_historical.parquet/` | `python main_static.py` |
| **Daily OOT** | Daily inference (90 days) | `bronze_flight_oot.parquet/` | `python main_static.py --snapshotdate 2025-01-01` |

### **Directory Structure**

```
datamart/bronze/flight/
├── bronze_flight_historical.parquet/   # Training data (Jan 2023 - Dec 2024)
│   ├── year_month=2023-01/
│   ├── year_month=2023-02/
│   └── ... (24 partitions)
│
└── bronze_flight_oot.parquet/         # OOT daily data (Jan-Mar 2025)
    ├── snapshot_date=2025-01-01/
    ├── snapshot_date=2025-01-02/
    └── ... (90 daily partitions)
```

### **Daily OOT Workflow**

#### **Step 1: Prepare OOT Data**

Place OOT CSV files in separate directory:
```
data/flight/oot/
├── T_ONTIME_REPORTING-01_25.csv  # January 2025
├── T_ONTIME_REPORTING-02_25.csv  # February 2025
└── T_ONTIME_REPORTING-03_25.csv  # March 2025
```

#### **Step 2: Process Daily Bronze**

```bash
# Process January 1, 2025
python main_static.py --snapshotdate 2025-01-01

# Process January 2, 2025
python main_static.py --snapshotdate 2025-01-02

# ... continue for 90 days
```

**Expected output per day**:
```
================================================================================
BRONZE LAYER PROCESSING - FLIGHT DELAY DATA
================================================================================

Mode: DAILY OOT
Processing date: 2025-01-01
Output path: datamart/bronze/flight/bronze_flight_oot.parquet

  Loading: T_ONTIME_REPORTING-01_25.csv
    Initial rows in CSV: 52,431
    Filtering for date: 2025-01-01
    After date filter: 1,683
    After NYC filter: 1,391

  Adding derived columns...
    ✓ year_month
    ✓ sort_time
    ✓ is_delayed_15
    ✓ DayOfWeek
    ✓ IsWeekend
    ✓ IsPublicHoliday
    ✓ processing_timestamp

SAVING TO PARQUET
  Output path: datamart/bronze/flight/bronze_flight_oot.parquet
  Partitioning by: snapshot_date
  Mode: overwrite
  ✓ Parquet file saved successfully!

VALIDATION CHECKS
1. Total Rows: 1,391
   ✓ PASS - Row count reasonable for daily data

2. Date Range: 2025-01-01 to 2025-01-01
   ✓ PASS - Single date as expected for daily data

✓ Processing time: 0.8 minutes
```

#### **Step 3: Daily Inference Pipeline**

```
For each day (Day 1 to Day 90):

  Bronze (1 day) 
      ↓
  Silver (features for 1 day)
      ↓
  Gold (merged with weather for 1 day)
      ↓
  Model Prediction (using trained model)
      ↓
  Compare vs True Label
      ↓
  Track Performance Metrics
```

#### **Step 4: Monthly Retraining (After 30 days)**

After January 2025 completes:
Manually merge OOT data into historical
# (Consolidate January 2025 daily partitions into monthly partition) and retrain

---

### **Performance Comparison**

| Mode | Processing Time | Output Size | Use Case |
|------|-----------------|-------------|----------|
| **Historical Batch** | ~15 minutes | 15 MB (1M rows) | One-time training data load |
| **Daily OOT** | <1 minute | ~10-50 KB (1K-2K rows) | Fast daily inference |

---

### **OOT Validation Metrics**

Track these metrics over 90 days:
- **Daily accuracy**
- **Daily precision/recall**
- **Delay prediction accuracy**
- **Model drift indicators**
- **Performance degradation**

---

### **Troubleshooting Daily OOT**

#### **Issue: CSV file not found**
```
ERROR: CSV file not found for date 2025-01-01: data/flight/oot/T_ONTIME_REPORTING-01_25.csv
```
**Solution**: Ensure OOT CSV files are in `data/flight/oot/` directory.

#### **Issue: No flights for date**
```
WARNING: No flights found for 2025-01-01
```
**Solution**: Check if the date exists in the CSV file. Some dates may have no NYC flights.

#### **Issue: Wrong date format**
```
ERROR: Invalid date format: 01-01-2025
```
**Solution**: Use format YYYY-MM-DD (e.g., 2025-01-01).

---

### **Command Reference**

```bash
# Historical batch (default)
python main_static.py

# Single day OOT
python main_static.py --snapshotdate 2025-01-01

# Help
python main_static.py --help

# Verify OOT Bronze
ls -lh datamart/bronze/flight/bronze_flight_oot.parquet/

# Count OOT partitions
ls datamart/bronze/flight/bronze_flight_oot.parquet/ | grep snapshot_date | wc -l
```

---

## 👥 Team Collaboration

### **Parallel Development**

**Flight Pipeline** (Your work):
- `utils/data_processing_flight_bronze.py`
- `main_static.py`
- Bronze output: `datamart/bronze/flight/`

**Weather Pipeline** (Colleague's work):
- `utils/data_processing_weather_bronze.py`
- Similar structure for weather data
- Bronze output: `datamart/bronze/weather/`

**Integration Point**: Gold layer (Days 5-7)

### **Git Workflow**

```bash
# Create feature branch
git checkout -b feature/flight-bronze

# Make changes, commit frequently
git add .
git commit -m "feat: implement Bronze layer with NYC filter"

# Push to remote
git push origin feature/flight-bronze

# Create Pull Request on GitHub
# Request review from teammate
```

### **Best Practices**
1. Work in separate feature branches
2. Communicate before modifying shared files
3. Use descriptive commit messages
4. Review each other's pull requests
5. Keep `main` branch stable

---

## 📦 Key Files Explained

### **`main_static.py`**
- Orchestrates entire Bronze processing pipeline
- Calls functions from `utils/data_processing_flight_bronze.py`
- Handles Spark initialization and shutdown
- Runs validation and prints summary

### **`utils/data_processing_flight_bronze.py`**
- Core Bronze layer logic
- Functions:
  - `process_single_csv()`: Load one month
  - `add_derived_columns()`: Create target, temporal features
  - `process_all_months_to_bronze()`: Main processing function
  - `validate_bronze_parquet()`: Data quality checks
  - `print_holiday_list()`: Display detected holidays

### **`test_setup.py`**
- Environment validation script
- Checks: Python, packages, Spark, data files, permissions
- Run before `main_static.py` to catch issues early

### **`docker-compose.yaml`**
- Defines Jupyter service
- Mounts local directory to `/app` in container
- Exposes port 8888 for Jupyter access

---

## 🎯 Success Criteria

✅ **Bronze Layer Complete When**:
- [x] 24 monthly CSVs processed successfully
- [x] Single Parquet file created
- [x] ~1M rows in output
- [x] 100% NYC metro coverage (JFK/LGA/EWR)
- [x] All derived columns present
- [x] Sorted by FlightDate and sort_time
- [x] Partitioned by year_month
- [x] All validation checks pass
- [x] Teammate can clone repo and reproduce results

---

## 📚 Additional Resources

**US DOT Data Source**: https://www.transtats.bts.gov/  
**PySpark Documentation**: https://spark.apache.org/docs/latest/api/python/  
**Python holidays library**: https://pypi.org/project/holidays/  

---

