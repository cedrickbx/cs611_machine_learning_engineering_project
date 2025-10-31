from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os

import json                                          # NEW
import pendulum                                      # NEW

# --------------------------------------------------------------------
# Global defaults
# --------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

SCRIPTS_DIR = "/opt/airflow/scripts"

# Scripts
## Bronze table store
FLIGHT_BRONZE_SCRIPT    = "bronze_flight_store.py"
WEATHER_SCRIPT          = "bronze_weather_store.py"
FORECAST_SCRIPT         = "bronze_forecast_store.py"
AIRPORT_BRONZE_SCRIPT   = "bronze_airport_store.py"

## Silver table store
AIRPORT_SILVER_SCRIPT   = "silver_airport_store.py"

# Writable directories inside the Airflow container
## Read Data
RAW_WEATHER_DIR         = "/opt/airflow/data/weather_history"
AIRPORTS_CSV            = "/opt/airflow/data/airports/airports.csv"
AIRPORT_FREQS_CSV       = "/opt/airflow/data/airports/airport-frequencies.csv"

## Store in datamart dir
WEATHER_PARQUET_DIR           = "/opt/airflow/datamart/bronze/weather_history"
WEATHER_FORECAST_OUT_DIR      = "/opt/airflow/datamart/bronze/forecast"
AIRPORT_BRONZE_DIR            = "/opt/airflow/datamart/bronze/airport"
AIRPORT_SILVER_DIR            = "/opt/airflow/datamart/silver/airport"

STATE_PATH = "/opt/airflow/datamart/gold/model_registry/last_training.json"
MIN_RETRAIN_DAYS = 60  # ~ every 2 months

# Helpers for retraining schedule
def _should_retrain() -> bool:
    """
    Return True if: (a) no history exists; or (b) last training >= MIN_RETRAIN_DAYS ago.
    Uses Asia/Singapore TZ for consistency with your environment.
    """
    now = pendulum.now("Asia/Singapore")
    if not os.path.exists(STATE_PATH):
        return True
    try:
        with open(STATE_PATH, "r") as f:
            meta = json.load(f)
        last_iso = meta.get("last_trained_at")
        if not last_iso:
            return True
        last = pendulum.parse(last_iso)
        return (now - last).days >= MIN_RETRAIN_DAYS
    except Exception:
        # If the marker is corrupted, be safe and retrain.
        return True

def _mark_trained():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"last_trained_at": pendulum.now("Asia/Singapore").to_iso8601_string()}, f)

def choose_daily_path_since_last(**_):
    # Always run inference; add retrain iff it's time.
    targets = ["daily_jobs.infer.entry"]
    if _should_retrain():
        targets.append("daily_jobs.retrain.entry")
    return targets


with DAG(
    dag_id="flight_delay_prediction_ML_pipeline",
    description="Branching: 2024-12-31 runs historical + airports once; otherwise run OOT daily.",
    start_date=datetime(2024, 12, 31),              # first run is the special day
    end_date=datetime(2025, 3, 31),
    schedule_interval="@daily",
    catchup=True,                                  # daily going forward
    max_active_runs=1,
    render_template_as_native_obj=True,
    default_args=default_args,
    tags=["bronze", "prep", "flight", "weather", "forecast"],
    params={
        #"run_hist": False,   # (kept for compatibility but branching now decides)
        #"run_oot": True,

        # Historical controls
        "hist_weather_download": True,   # set False to skip raw downloads

        # OOT controls
        "oot_start": "{{ ds }}",   # e.g., 2025-10-30
        "oot_end":   "{{ ds }}",   # same day by default
        "oot_cycles": "0,12",
        "oot_fhours": "6,12,24,48,72",

        # Airports Params controls
        "airport_freq_types": "TWR,APP,A/D,ATIS,AWOS,GND",
        "airport_scheduled_only": "true",
        "airport_subset_iata": "JFK,LGA,EWR",
    },
) as dag:

    start = EmptyOperator(task_id="start")
    done  = EmptyOperator(task_id="done", trigger_rule=TriggerRule.ALL_DONE)

    # ==========================================================
    # BRANCH DECISION — 2024-12-31 => historical + airports; else OOT
    # ==========================================================
    def choose_branch(**context):
        # Use the logical (scheduled) date in YYYY-MM-DD
        ds = context["ds"]  # e.g. "2024-12-31"
        if ds == "2024-12-31":
            # run both data_preprocessing_batch and airports_ref
            return ["one_time_batch.entry"]
        else:
            # run oot branch only
            return ["daily_jobs.entry"]

    branch = BranchPythonOperator(
        task_id="one_time_or_daily",
        python_callable=choose_branch,
    )

    # ===============================
    # Group 1: ONE-TIME BATCH (2024-12-31)
    # ===============================
    with TaskGroup(group_id="one_time_batch") as tg_one_time:

        entry = EmptyOperator(task_id="entry")

        # ==========================================================
        # BRANCH A — HISTORICAL (one-shot): Flight + Weather 2023–2024
        # ==========================================================
        with TaskGroup(group_id="data_preprocessing_batch") as tg_dp_batch:

            hist_entry = EmptyOperator(task_id="entry")

            # 1A) Ensure raws 2023–2025 exist; also write historical parquet (2023–2024)
            #    --download-data makes the script fetch NOAA ISD CSVs if missing.
            weather_bronze = BashOperator(
                task_id="weather_bronze",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {RAW_WEATHER_DIR} {WEATHER_PARQUET_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    # --download-data guarded by param
                    "EXTRA='' ; "
                    "if [ '{{ params.hist_weather_download }}' = 'True' ]; then EXTRA='--download-data'; fi ; "
                    f"python3 {WEATHER_SCRIPT} $EXTRA"
                ),
                env={
                    "NOAA_DATA_DIR": RAW_WEATHER_DIR,
                    "WEATHER_PARQUET_DIR": WEATHER_PARQUET_DIR,
                },
            )

            # Flight historical: no --snapshotdate to run the full batch (2023–2024) as your script supports
            flight_bronze = BashOperator(
                task_id="flight_bronze",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FLIGHT_BRONZE_SCRIPT}"
                ),
            )

            # =============================================================
            # BRANCH C — AIRPORTS (reference) — Bronze → Silver (one-shot)
            # =============================================================
            # What it does:
            #   - Ingest two static reference CSVs (airports & airport_frequencies)
            #     into Bronze as Parquet (no business transforms, metadata columns only).
            #   - Build the Silver-level canonical airport dimension (US-wide wide table
            #     with has_* flags; optional IATA subset if your script supports).
            #
            # Why here & when to run:
            #   - Airports are static/slow-moving reference data → not part of OOT.
            #   - Run once (or ad-hoc when the CSVs update). Downstream facts (Flight/Weather)
            #     can join this dimension later in Silver/Gold.
            # =============================================================

            # --- Short Circuit: skip airports pipeline if airport_silver already exists ---
            airports_ready = ShortCircuitOperator(
                task_id="airports_ready",
                python_callable=lambda: not os.path.exists(AIRPORT_SILVER_DIR),
            )

            # # 1) Bronze load (two CSVs → two Parquet folders):
            # #    The Bronze script should internally call your process_bronze_airports_table(...)
            # airports_bronze = BashOperator(
            #     task_id="airports_bronze",
            #     bash_command=(
            #         "set -euxo pipefail && "
            #         f"mkdir -p {AIRPORT_BRONZE_DIR} && "
            #         f"cd {SCRIPTS_DIR} && "
                    
            #         # First: airports.csv
            #         f"python3 {AIRPORT_BRONZE_SCRIPT} "
            #         f"--csv '{AIRPORTS_CSV}' "
            #         f"--bronze-dir '{AIRPORT_BRONZE_DIR}' "
            #         "--table airports --mode overwrite && "
                    
            #         # Second: airport-frequencies.csv
            #         f"python3 {AIRPORT_BRONZE_SCRIPT} "
            #         f"--csv '{AIRPORT_FREQS_CSV}' "
            #         f"--bronze-dir '{AIRPORT_BRONZE_DIR}' "
            #         "--table airport_frequencies --mode overwrite"
            #     ),
            # )

            airports_bronze_airports = BashOperator(
                task_id="airports_bronze_airports",
                bash_command=(
                    "set -euxo pipefail\n"
                    f"test -f '{AIRPORTS_CSV}' || (echo 'MISSING: {AIRPORTS_CSV}' && exit 1)\n"
                    f"mkdir -p '{AIRPORT_BRONZE_DIR}'\n"
                    f"cd '{SCRIPTS_DIR}'\n"
                    f"python3 {AIRPORT_BRONZE_SCRIPT} "
                    f"--csv '{AIRPORTS_CSV}' "
                    f"--bronze-dir '{AIRPORT_BRONZE_DIR}' "
                    "--table airports --mode overwrite\n"
                ),
            )

            airports_bronze_freqs = BashOperator(
                task_id="airports_bronze_freqs",
                bash_command=(
                    "set -euxo pipefail\n"
                    f"test -f '{AIRPORT_FREQS_CSV}' || (echo 'MISSING: {AIRPORT_FREQS_CSV}' && exit 1)\n"
                    f"mkdir -p '{AIRPORT_BRONZE_DIR}'\n"
                    f"cd '{SCRIPTS_DIR}'\n"
                    f"python3 {AIRPORT_BRONZE_SCRIPT} "
                    f"--csv '{AIRPORT_FREQS_CSV}' "
                    f"--bronze-dir '{AIRPORT_BRONZE_DIR}' "
                    "--table airport_frequencies --mode overwrite\n"
                ),
            )

            # 2) Silver build (canonicalized wide dim from Bronze):
            #    Reads Bronze Parquet (airports, airport_frequencies), produces
            #    - silver/airport/US_airports/ (wide dim with arrays + has_* flags)
            #    - optional IATA subset (e.g., silver/airport/new_york/) if your script supports --iata
            airports_silver = BashOperator(
                task_id="airports_silver",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {AIRPORT_SILVER_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {AIRPORT_SILVER_SCRIPT} "
                    f"--bronze-dir '{AIRPORT_BRONZE_DIR}' "
                    f"--silver-dir '{AIRPORT_SILVER_DIR}'"
                    # Append if needed:
                    # " --iata 'JFK,LGA,EWR'"
                ),
            )

            # Run both in parallel; if you prefer sequencing, chain them
            hist_entry >> [weather_bronze, flight_bronze, airports_ready]
            # Bronze → Silver for Airports
            # airports_bronze >> airports_silver
            airports_ready >> airports_bronze_airports >> airports_bronze_freqs >> airports_silver

        data_dep_check_batch = EmptyOperator(task_id="data_dep_check_batch")

        entry >> tg_dp_batch >> data_dep_check_batch

        # ===============================
        # Initial training (one-time, after batch data preprocessing)
        # ===============================
        with TaskGroup(group_id="initial_train") as initial_train:

            initial_training_entry = EmptyOperator(task_id="entry")
            model_training_initial = EmptyOperator(task_id="model_training_initial")
            model_promotion_initial = EmptyOperator(task_id="model_promotion_initial")

        data_dep_check_batch >> initial_training_entry >> model_training_initial >> model_promotion_initial

    # ===============================
    # Group 2: DAILY JOBS (all other days)
    # ===============================
    with TaskGroup(group_id="daily_jobs") as tg_daily:

        entry = EmptyOperator(task_id="entry")

        # ==========================================================
        # BRANCH B — OOT RANGE (separate branch): Weather + Forecast
        # ==========================================================
        with TaskGroup(group_id="data_preprocessing_daily") as tg_dp_daily:

            dp_daily_entry = EmptyOperator(task_id="dp_daily_entry")

            flight_bronze_daily = BashOperator(
                task_id="flight_bronze_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    "while read -r DS; do "
                    f"  python3 {FLIGHT_BRONZE_SCRIPT} --snapshotdate \"$DS\"; "
                    "done < <(python3 - <<'PY'\n"
                    "from datetime import datetime, timedelta\n"
                    "s='{{ ds }}'.strip()\n"
                    "e='{{ ds }}'.strip()\n"
                    "S=datetime.fromisoformat(s)\n"
                    "E=datetime.fromisoformat(e)\n"
                    "if E < S: S, E = E, S  # swap if user inverted\n"
                    "d=S\n"
                    "while d <= E:\n"
                    "    print(d.strftime('%Y-%m-%d'))\n"
                    "    d += timedelta(days=1)\n"
                    "PY\n"
                    ")"
                ),
            )

            '''flight_oot_range = BashOperator(
                task_id="flight_oot_range",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    "while read -r DS; do "
                    f"  python3 {FLIGHT_BRONZE_SCRIPT} --snapshotdate \"$DS\"; "
                    "done < <(python3 - <<'PY'\n"
                    "from datetime import datetime, timedelta\n"
                    "s='{{ params.oot_start }}'.strip()\n"
                    "e='{{ params.oot_end | default(params.oot_start) }}'.strip()\n"
                    "S=datetime.fromisoformat(s)\n"
                    "E=datetime.fromisoformat(e)\n"
                    "if E < S: S, E = E, S  # swap if user inverted\n"
                    "d=S\n"
                    "while d <= E:\n"
                    "    print(d.strftime('%Y-%m-%d'))\n"
                    "    d += timedelta(days=1)\n"
                    "PY\n"
                    ")"
                ),
            )'''

            # 1B) Backfill OOT weather history for a date RANGE (named file per day)
            # Read dates from dag_run.conf; fall back to sensible defaults if not provided.
            # This calls your existing writer in 'range' mode and skips re-writing historical.
            weather_bronze_daily = BashOperator(
                task_id="weather_bronze_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {RAW_WEATHER_DIR} {WEATHER_PARQUET_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {WEATHER_SCRIPT} "
                    "--snapshotdate '{{ ds }}' "
                    "--no-hist --download-data"
                ),
                env={
                    "NOAA_DATA_DIR": RAW_WEATHER_DIR,
                    "WEATHER_PARQUET_DIR": WEATHER_PARQUET_DIR,
                },
            )

            '''weather_oot_range = BashOperator(
                task_id="weather_oot_range",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {RAW_WEATHER_DIR} {WEATHER_PARQUET_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {WEATHER_SCRIPT} "
                    "--oot-start '{{ params.oot_start }}' "
                    "--oot-end   '{{ params.oot_end | default(params.oot_start) }}' "
                    "--no-hist --download-data"
                ),
                env={
                    "NOAA_DATA_DIR": RAW_WEATHER_DIR,
                    "WEATHER_PARQUET_DIR": WEATHER_PARQUET_DIR,
                },
            )'''

            # 2A) Backfill OOT forecasts for the SAME date range (per valid_date file)
            # You can trim cycles/leads for speed; these are balanced defaults.
            '''forecast_oot_range = BashOperator(
                task_id="forecast_oot_range",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {WEATHER_FORECAST_OUT_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FORECAST_SCRIPT} "
                    "--oot-start '{{ params.oot_start }}' "
                    "--oot-end   '{{ params.oot_end | default(params.oot_start) }}' "
                    "--cycles {{ params.oot_cycles }} "
                    "--fhours {{ params.oot_fhours }} "
                    f"--out {WEATHER_FORECAST_OUT_DIR} "
                    "--no-hist"
                ),
            )'''

            # Run all three OOT tasks in parallel (after the group-level gate/deps)
            #weather_bronze >> weather_oot_range

            # Parallel OOT: weather + forecast
            #oot_entry >> [weather_oot_range, forecast_oot_range, flight_oot_range]
            dp_daily_entry >> [weather_bronze_daily, flight_bronze_daily]

        # daily_jobs wiring
        entry >> tg_dp_daily

        # ===============================
        # Branch: choose retraining+inference+monitoring (every two months) vs inference+monitoring (daily)
        # ===============================
        def choose_daily_path(**context):
            # Run the model pipeline daily.
            # Retraining happens every 2 months (Jan 2025 as base month => months_diff % 2 == 0).
            ds = context["ds"]  # 'YYYY-MM-DD'
            y, m, d = map(int, ds.split("-"))

            base_year, base_month = 2025, 1
            months_diff = (y - base_year) * 12 + (m - base_month)

            targets = ["daily_jobs.infer.entry"]  # inference+monitoring every month
            if months_diff % 2 == 0:
                targets.append("daily_jobs.retrain.entry")  # retrain every 2 months
            return targets

        daily_branch = BranchPythonOperator(
            task_id="retrain_or_infer",
            python_callable=choose_daily_path,
        )

        data_dep_check_daily = EmptyOperator(task_id="data_dep_check_daily")

        # ---- Retraining branch (same sequence as the initial training) ----
        with TaskGroup(group_id="retrain") as tg_retrain:
            retrain_entry = EmptyOperator(task_id="entry")
            model_training_retrain = EmptyOperator(task_id="model_training_retrain")
            model_promotion_retrain = EmptyOperator(task_id="model_promotion_retrain")
            retrain_entry >> model_training_retrain >> model_promotion_retrain

        # ---- Inference branch (daily) ----
        with TaskGroup(group_id="infer") as tg_infer:
            infer_entry = EmptyOperator(
                task_id="entry",
                trigger_rule=TriggerRule.ONE_SUCCESS  # ← run if either upstream succeeds
            )
            model_inferencing = EmptyOperator(task_id="model_inferencing")
            infer_entry >> model_inferencing

        # ---- Monitoring branch (daily) ----
        with TaskGroup(group_id="monitor") as tg_monitor:
            monitor_entry = EmptyOperator(task_id="entry")
            model_monitoring = EmptyOperator(task_id="model_monitoring")
            monitor_entry >> model_monitoring

        tg_infer >> tg_monitor
        tg_dp_daily >> data_dep_check_daily >> daily_branch >> [tg_retrain, tg_infer]
        tg_retrain >> tg_infer

    # -------- Orchestration with branching --------
    start >> branch
    branch >> tg_one_time
    branch >> tg_daily
    [tg_one_time, tg_daily] >> done