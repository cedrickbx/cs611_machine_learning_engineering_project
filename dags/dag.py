from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os

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

with DAG(
    dag_id="bronze_data_preparation_pipeline",
    description="Batch Bronze: Historical Flight + Weather (2023–2024) and separate OOT branch for Weather/Forecast.",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,              # manual trigger
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
    default_args=default_args,
    tags=["bronze", "prep", "flight", "weather", "forecast"],
    params={
        "run_hist": False,
        "run_oot": True,
        
        # Historical controls (you can widen if scripts support it)
        "hist_weather_download": True,   # set False to skip raw downloads
        
        # OOT controls
        "oot_start": "2025-01-01",
        "oot_end":   "2025-01-01",
        #"oot_end":   "2025-03-31",
        "oot_cycles": "0,12",
        "oot_fhours": "6,12,24,48,72",

        # Airports Params controls
        "airport_freq_types": "TWR,APP,A/D,ATIS,AWOS,GND",  # set airports facilities
        "airport_scheduled_only": "true",
        "airport_subset_iata": "JFK,LGA,EWR",               # selected airports 
    },
) as dag:

    start = EmptyOperator(task_id="start")
    done  = EmptyOperator(task_id="done", trigger_rule=TriggerRule.ALL_DONE)

    # ==========================================================
    # BRANCH A — HISTORICAL (one-shot): Flight + Weather 2023–2024
    # ==========================================================
    with TaskGroup(group_id="historical_bronze") as tg_hist:

        # 1A) Ensure raws 2023–2025 exist; also write historical parquet (2023–2024)
        #    --download-data makes the script fetch NOAA ISD CSVs if missing.
        weather_hist = BashOperator(
            task_id="weather_hist",
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
        flight_hist = BashOperator(
            task_id="flight_hist",
            bash_command=(
                "set -euxo pipefail && "
                f"cd {SCRIPTS_DIR} && "
                f"python3 {FLIGHT_BRONZE_SCRIPT}"
            ),
        )

        # Run both in parallel; if you prefer sequencing, chain them
        [weather_hist, flight_hist]

    # ==========================================================
    # BRANCH B — OOT RANGE (separate branch): Weather + Forecast
    # ==========================================================
    # ------------------ OOT BRANCH (MANUAL GATE) ------------------
    # Gate: only run OOT when params.run_oot == True
    def flag(ctx_key, default=False, **context):
        v = context["params"].get(ctx_key, default)
        return str(v).strip().lower() in {"1","true","t","yes","y","on"}

    hist_gate = ShortCircuitOperator(
        task_id="hist_gate",
        python_callable=lambda **ctx: flag("run_hist", True, **ctx),
    )

    oot_gate = ShortCircuitOperator(
        task_id="oot_gate",
        python_callable=lambda **ctx: flag("run_oot", False, **ctx),
    )
    
    with TaskGroup(group_id="oot_bronze") as tg_oot:

        flight_oot_range = BashOperator(
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
        )

        # 1B) Backfill OOT weather history for a date RANGE (named file per day)
        # Read dates from dag_run.conf; fall back to sensible defaults if not provided.
        # This calls your existing writer in 'range' mode and skips re-writing historical.
        weather_oot_range = BashOperator(
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
        )

        # 2A) Backfill OOT forecasts for the SAME date range (per valid_date file)
        # You can trim cycles/leads for speed; these are balanced defaults.
        forecast_oot_range = BashOperator(
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
        )

        # Run all three OOT tasks in parallel (after the group-level gate/deps)
        #weather_hist >> weather_oot_range

        # Parallel OOT: weather + forecast
        [weather_oot_range, forecast_oot_range, flight_oot_range]

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

    with TaskGroup(group_id="airports_ref") as tg_airports:

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

        # Bronze → Silver for Airports
        # airports_bronze >> airports_silver
        airports_ready >> airports_bronze_airports >> airports_bronze_freqs >> airports_silver


    # -------- Orchestration with dependency --------
    # Start -> HIST (weather & flight in parallel)
    start >> hist_gate >> tg_hist
    start >> oot_gate >> tg_oot
    start >> tg_airports >> [tg_hist, tg_oot]

    # Finish when HIST (both tasks) and OOT (both tasks) are done
    [tg_hist, tg_oot] >> done