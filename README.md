
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