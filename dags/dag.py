from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os
import mlflow

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
WEATHER_BRONZE_SCRIPT   = "bronze_weather_store.py"
# FORECAST_SCRIPT         = "bronze_forecast_store.py"
AIRPORT_BRONZE_SCRIPT   = "bronze_airport_store.py"

## Silver table store
AIRPORT_SILVER_SCRIPT   = "silver_airport_store.py"
WEATHER_SILVER_SCRIPT   = "silver_weather_store.py"
FLIGHT_SILVER_SCRIPT    = "silver_flight_store.py"

# Gold table store
FLIGHT_GOLD_SCRIPT      = "gold_flight_store.py"
FEATURE_GOLD_SCRIPT     = "gold_combined_store.py"

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

FEATURE_GOLD_DIR = "/opt/airflow/datamart/gold/combined"
HIST_COMBINED_PARQUET = "/opt/airflow/datamart/gold/combined/gold_combined_historical.parquet/snapshot_date=2023-01-01"
PRED_DIR = "/opt/airflow/datamart/gold/model_predictions"


STATE_PATH = "/opt/airflow/datamart/gold/model_registry/last_training.json"
MIN_RETRAIN_DAYS = 60  # ~ every 2 months

# Helpers for retraining schedule
def choose_hist_or_train(**context):
    """
    If the historical combined parquet exists, skip preprocessing and go straight to initial training.
    Otherwise, run the full one-time batch preprocessing.
    """
    exists = os.path.exists(HIST_COMBINED_PARQUET)
    if exists:
        # jump straight to initial training entry task inside the one_time_batch group
        return ["one_time_batch.initial_train.entry"]
    else:
        # run the usual data preprocessing batch
        return ["one_time_batch.data_preprocessing_batch.entry"]

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
    tags=["bronze", "silver", "gold", "prep", "flight", "weather", "forecast"],
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
                    f"python3 {WEATHER_BRONZE_SCRIPT} $EXTRA"
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

            # =====================================================
            # 🚀 NEW: SILVER + GOLD LAYER PIPELINE
            # =====================================================

            # 1B) Silver Weather (runs after Bronze Weather)
            weather_silver = BashOperator(
                task_id="weather_silver",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {WEATHER_SILVER_SCRIPT}"
                ),
            )

            # 1C) Silver Flight (runs after Bronze Flight)
            flight_silver = BashOperator(
                task_id="flight_silver",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FLIGHT_SILVER_SCRIPT}"
                ),
            )

            # 1D) Gold Flight (after both Silver Flight + Silver Weather)
            flight_gold = BashOperator(
                task_id="flight_gold",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FLIGHT_GOLD_SCRIPT}"
                ),
            )

            # 1E) Gold Combined Features (after Gold Flight + Silver Weather)
            feature_gold = BashOperator(
                task_id="feature_gold",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FEATURE_GOLD_SCRIPT}"
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

            # =====================================================
            # DEPENDENCIES
            # =====================================================
            # Run both in parallel; if you prefer sequencing, chain them
            hist_entry >> [weather_bronze, flight_bronze, airports_ready]

            # Bronze → Silver → Gold → Combined
            weather_bronze >> weather_silver
            flight_bronze >> flight_silver >> flight_gold
            [flight_gold, weather_silver] >> feature_gold

            airports_ready >> airports_bronze_airports >> airports_bronze_freqs >> airports_silver

        data_dep_check_batch = EmptyOperator(
            task_id="data_dep_check_batch",
            trigger_rule=TriggerRule.ONE_SUCCESS,
        )

        # ===============================
        # Initial training (one-time, after batch data preprocessing)
        # ===============================
        with TaskGroup(group_id="initial_train") as initial_train:

            initial_training_entry = EmptyOperator(task_id="entry")
            model_training_initial = BashOperator(
                task_id="model_training_initial",
                bash_command=(
                    "set -euxo pipefail\n"
                    # Assert the training script exists
                    f"test -f '{SCRIPTS_DIR}/train_xgboost.py' || "
                    f"(echo 'Missing: {SCRIPTS_DIR}/train_xgboost.py' >&2; exit 1)\n"
                    # Run with ABSOLUTE PATH (not relative)
                    f"python3 '{SCRIPTS_DIR}/train_xgboost.py' "
                    "--hardcode-q1-2023 "
                    "--data-path \"$HIST_COMBINED_PARQUET\" "
                    "--date-col {{ var.value.get('date_col', 'FlightDate') }} "
                    "--label-col {{ var.value.get('label_col', 'IS_DELAYED') }} "
                    "--total-months {{ var.value.get('total_months', 24) }} "
                    "--val-months {{ var.value.get('val_months', 3) }} "
                    "--test-months {{ var.value.get('test_months', 3) }} "
                    "--n-trials {{ var.value.get('n_trials', 3) }} "
                    "--mlflow-experiment {{ var.value.get('mlflow_experiment', 'flight_delay_training') }} "
                    "--outdir /tmp\n"
                ),
                env={
                    "HIST_COMBINED_PARQUET": HIST_COMBINED_PARQUET,  # e.g. "/opt/airflow/datamart/gold/combined/gold_combined_historical_2023_01_01_2024_12_31.parquet"
                    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                    "EXPERIMENT_DT": "{{ ds }}",   # Airflow logical date (e.g. 2025-11-03)
                },
                append_env=True,
            )
            '''model_training_initial = BashOperator(
                task_id="model_training_initial",
                bash_command=(
                    "python scripts/train_xgboost.py "
                    "--data-path {HIST_COMBINED_PARQUET} "
                    "--date-col {{ var.value.date_col | default('FlightDate') }} "
                    "--label-col {{ var.value.label_col | default('IS_DELAYED') }} "
                    "--exclude-cols \"{{ var.value.exclude_cols | default('weather_ceiling_code,weather_visibility_var_code_idx,weather_wind_dir_deg,weather_visibility_var_code') }}\" "
                    "--total-months {{ var.value.total_months | default(24) }} "
                    "--val-months {{ var.value.val_months | default(3) }} "
                    "--test-months {{ var.value.test_months | default(3) }} "
                    "--n-trials {{ var.value.n_trials | default(60) }} "
                    "--mlflow-experiment {{ var.value.mlflow_experiment | default('flight_delay_xgb_optuna_ord3') }} "
                    "--outdir /tmp"
                ),
                env={
                    "MLFLOW_TRACKING_URI": "{{ var.value.mlflow_uri }}",
                },
                append_env=True,
            )'''
            model_promotion_initial = EmptyOperator(task_id="model_promotion_initial")

        data_dep_check_batch >> initial_training_entry >> model_training_initial >> model_promotion_initial

        # ------------------------
        # branch to skip or run preprocessing
        # ------------------------
        skip_or_preprocess_hist = BranchPythonOperator(
            task_id="skip_or_preprocess_hist",
            python_callable=choose_hist_or_train,
        )

        # wire the new branch
        entry >> skip_or_preprocess_hist
        # If preprocessing is chosen, go into that group’s entry (then later to data_dep_check_batch → initial_train)
        skip_or_preprocess_hist >> tg_dp_batch >> data_dep_check_batch
        # If skip is chosen, jump straight to initial training entry
        skip_or_preprocess_hist >> initial_training_entry

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

            # 1B) Backfill OOT weather history for a date RANGE (named file per day)
            # Read dates from dag_run.conf; fall back to sensible defaults if not provided.
            # This calls your existing writer in 'range' mode and skips re-writing historical.
            '''weather_bronze_daily = BashOperator(
                task_id="weather_bronze_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"mkdir -p {RAW_WEATHER_DIR} {WEATHER_PARQUET_DIR} && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {WEATHER_BRONZE_SCRIPT} "
                    "--snapshotdate '{{ ds }}' "
                    "--no-hist --download-data"
                ),
                env={
                    "NOAA_DATA_DIR": RAW_WEATHER_DIR,
                    "WEATHER_PARQUET_DIR": WEATHER_PARQUET_DIR,
                },
            )'''

            # 2) Silver
            # Weather silver: current script processes full bronze → silver (idempotent)
            '''weather_silver_daily = BashOperator(
                task_id="weather_silver_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {WEATHER_SILVER_SCRIPT}"
                ),
            )'''

            # Flight silver: per-day slice
            flight_silver_daily = BashOperator(
                task_id="flight_silver_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FLIGHT_SILVER_SCRIPT} --snapshotdate '{{{{ ds }}}}'"
                ),
            )

            # 3) Gold (Flight)
            flight_gold_daily = BashOperator(
                task_id="flight_gold_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FLIGHT_GOLD_SCRIPT} --snapshotdate '{{{{ ds }}}}'"
                ),
            )

            # 4) Gold Combined (Flight ⨯ Weather)
            feature_gold_daily = BashOperator(
                task_id="feature_gold_daily",
                bash_command=(
                    "set -euxo pipefail && "
                    f"cd {SCRIPTS_DIR} && "
                    f"python3 {FEATURE_GOLD_SCRIPT} --snapshotdate '{{{{ ds }}}}'"
                ),
            )

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

            # ---------- dependencies ----------
            
            dp_daily_entry >> flight_bronze_daily >> flight_silver_daily >> flight_gold_daily >> feature_gold_daily
            # dp_daily_entry >> [weather_bronze_daily, flight_bronze_daily]
            # weather_bronze_daily >> weather_silver_daily 
            # [flight_gold_daily, weather_silver_daily] >> feature_gold_daily

        # Gate after features are ready
        data_dep_check_daily = EmptyOperator(task_id="data_dep_check_daily")

        # Decide retrain (bi-monthly) + always run infer
        def choose_daily_path(**context):

            ds = context["ds"]
            current_date = datetime.strptime(ds, "%Y-%m-%d").date()
            client = mlflow.tracking.MlflowClient("http://mlflow:5000")
            model_name = "Registered_FL_Delay_Model"

            try:
                try:
                    model_version = client.get_model_version_by_alias(model_name, "production")
                except Exception:
                    model_version = client.get_model_version_by_alias(model_name, "production")

                if not model_version or not getattr(model_version, "run_id", None):
                    print("[choose_daily_path] No valid production model, retrain.")
                    return ["daily_jobs.retrain.entry"]

                mv_tags = getattr(model_version, "tags", {})
                print(f"[choose_daily_path] Model version tags: {mv_tags}")
                linked_run_id = mv_tags.get("linked_run_id")
                print(f"[choose_daily_path] linked_run_id: {linked_run_id}")

                if not linked_run_id:
                    print("[choose_daily_path] No linked_run_id tag found → retrain.")
                    return ["daily_jobs.retrain.entry"]
        
                run = client.get_run(model_version.run_id)
                exp_dt_str = run.data.tags.get("EXPERIMENT_DT")
                if not exp_dt_str:
                    print("[choose_daily_path] Missing EXPERIMENT_DT, retrain.")
                    return ["daily_jobs.retrain.entry"]

                prod_date = datetime.strptime(exp_dt_str, "%Y-%m-%d").date()
                months_diff = (current_date.year - prod_date.year) * 12 + (current_date.month - prod_date.month)
                print(f"[choose_daily_path] Prod {prod_date}, now {current_date}, months_diff={months_diff}")
                return ["daily_jobs.retrain.entry"] if months_diff >= 2 else ["daily_jobs.infer.entry"]

            except Exception as e:
                print(f"[choose_daily_path] Error fetching model info: {e}")
                return ["daily_jobs.retrain.entry"]

        daily_branch = BranchPythonOperator(
            task_id="retrain_or_infer",
            python_callable=choose_daily_path,
        )

        # ---- Retraining branch (same sequence as the initial training) ----
        with TaskGroup(group_id="retrain") as tg_retrain:
            retrain_entry = EmptyOperator(task_id="entry")

            model_training_retrain = BashOperator(
                task_id="model_training_retrain",
                bash_command=(
                    "set -euxo pipefail\n"
                    # Verify the retrain script exists
                    f"test -f '{SCRIPTS_DIR}/train_xgboost.py' || "
                    f"(echo 'Missing: {SCRIPTS_DIR}/train_xgboost.py' >&2; exit 1)\n"
                    # Run retraining with dynamically set retrain date
                    f"python3 '{SCRIPTS_DIR}/train_xgboost.py' "
                    "--data-path /opt/airflow/datamart/gold/combined "
                    "--retrain-date {{ ds }} "
                    "--date-col {{ var.value.get('date_col', 'FlightDate') }} "
                    "--label-col {{ var.value.get('label_col', 'IS_DELAYED') }} "
                    "--total-months {{ var.value.get('total_months', 24) }} "
                    "--val-months {{ var.value.get('val_months', 3) }} "
                    "--test-months {{ var.value.get('test_months', 3) }} "
                    "--n-trials {{ var.value.get('n_trials', 3) }} "
                    "--mlflow-experiment {{ var.value.get('mlflow_experiment', 'flight_delay_training') }} "
                    "--outdir /tmp\n"
                ),
                env={
                    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                    "EXPERIMENT_DT": "{{ ds }}",     # Airflow execution date (e.g. 2025-03-01)
                    "DATAMART_ROOT": "/opt/airflow/datamart",
                },
            )

            model_promotion_retrain = BashOperator(
                task_id="model_promotion_retrain",
                bash_command=(
                    "set -euxo pipefail\n"
                    "echo 'Checking if retrained model outperforms current production model...'\n"
                    "echo 'Promotion logic placeholder (compare test_roc_auc_macro_ovr, test_f1_macro, etc.)'\n"
                ),
                env={
                    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                },
            )

            retrain_entry >> model_training_retrain >> model_promotion_retrain

        # ---- Inference branch (daily) ----
        with TaskGroup(group_id="infer") as tg_infer:
            infer_entry = EmptyOperator(
                task_id="entry",
                trigger_rule=TriggerRule.ONE_SUCCESS  # ← run if either upstream succeeds
            )

            model_inferencing = BashOperator(
                task_id="model_inferencing",
                bash_command=(
                    "set -euxo pipefail\n"
                    f"cd {SCRIPTS_DIR}\n"
                    f"python3 model_infer_monitor.py "
                    "--mode inference "
                    "--snapshotdate '{{ ds }}' "
                    f"--gold-feature-dir '{FEATURE_GOLD_DIR}' "
                    "--mlflow-experiment flight_delay_inference_monitoring "
                    "--model-name Registered_FL_Delay_Model "
                    "--model-alias production "
                    "--tracking-uri http://mlflow:5000 "
                    f"--pred-dir '{PRED_DIR}' "
                    "--write-local --write-csv"
                ),
                env={
                    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                    "EXPERIMENT_DT": "{{ ds }}",  # logged date for MLflow
                },
                append_env=True,
            )

            infer_entry >> model_inferencing

        # ---- Monitoring branch (daily) ----
        with TaskGroup(group_id="monitor") as tg_monitor:
            monitor_entry = EmptyOperator(task_id="entry")

            model_monitoring = BashOperator(
                task_id="model_monitoring",
                bash_command=(
                    "set -euxo pipefail\n"
                    f"cd {SCRIPTS_DIR}\n"
                    f"python3 model_infer_monitor.py "
                    "--mode monitor "
                    "--snapshotdate '{{ ds }}' "
                    f"--gold-feature-dir '{FEATURE_GOLD_DIR}' "
                    f"--pred-dir '{PRED_DIR}' "
                    "--mlflow-experiment flight_delay_inference_monitoring "
                    "--tracking-uri http://mlflow:5000"
                ),
                env={
                    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                    "EXPERIMENT_DT": "{{ ds }}",
                },
                append_env=True,
            )

            monitor_entry >> model_monitoring

        # wiring
        entry >> tg_dp_daily
        tg_dp_daily >> data_dep_check_daily >> daily_branch >> [tg_retrain, tg_infer]
        tg_retrain >> tg_infer >> tg_monitor

    # -------- Orchestration with branching --------
    start >> branch
    branch >> tg_one_time
    branch >> tg_daily
    [tg_one_time, tg_daily] >> done