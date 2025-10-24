
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
cs611_machine_learning_engineering_project/
├── README.md
├── .gitignore
├── docker-compose.yaml                 # Airflow + Jupyter services
├── Dockerfile                          # PySpark + Jupyter image
├── requirements.txt                    # Python deps
│
├── dags/                               # Airflow DAGs (mounted into scheduler/webserver)
│   ├── __init__.py
│   └── day.py
│
├── scripts/
│   ├── utils/                          # Shared utility functions
│   │   ├── __init__.py
│   │   └── data_processing_flight_bronze.py  # Bronze layer logic
│   │
│   ├── bronze_label_store.py           # Template task for Airflow (reference)
│   ├── main_static.py                  # Main Bronze processing script
│   ├── test_setup.py                   # Environment validation
│   └── notebooks/                      # Jupyter notebooks for EDA
│
├── data/                               # Raw input data (NOT tracked)
│   └── flight/
│       └── train/                      # 24 monthly CSVs
│           ├── T_ONTIME_REPORTING-01_23.csv
│           ├── T_ONTIME_REPORTING-02_23.csv
│           └── ... (22 more files)
│
└── datamart/                           # Pipeline outputs (NOT tracked)
    └── bronze/
        └── flight/
            └── bronze_flight_combined.parquet/   # ← Output
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