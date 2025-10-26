from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator, EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

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
FLIGHT_BRONZE_SCRIPT = "bronze_flight_store.py"
weather_history_bronze_script = "bronze_weather_store.py"
forecast_script = "bronze_forecast_store.py"

with DAG(
    dag_id="stage_training_and_oot_weather_forecast",
    description="One-off: stage training (2023-2024) + initial OOT (range) for weather & forecast Bronze",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,     # manual trigger only
    catchup=False,              # no auto backfills
    default_args=default_args,
    tags=["bronze", "staging", "backfill", "weather", "forecast"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ------------------------------
    # Group 1: Weather history
    # ------------------------------
    with TaskGroup(group_id="weather_history") as tg_weather:

        # 1A) Ensure raws 2023–2025 exist; also write historical parquet (2023–2024)
        #    --download-data makes the script fetch NOAA ISD CSVs if missing.
        weather_download_and_hist = BashOperator(
            task_id="weather_download_and_hist",
            bash_command=(
                "set -euo pipefail && "
                f"cd {SCRIPTS_DIR} && "
                # Writes historical parquet and downloads raws 2023–2025 if needed
                f"python3 {weather_history_bronze_script} --download-data"
            ),
        )

        # 1B) Backfill OOT weather history for a date RANGE (named file per day)
        # Read dates from dag_run.conf; fall back to sensible defaults if not provided.
        # This calls your existing writer in 'range' mode and skips re-writing historical.
        weather_oot_range = BashOperator(
            task_id="weather_oot_range",
            bash_command=(
                "set -euo pipefail && "
                f"cd {SCRIPTS_DIR} && "
                f"python3 {weather_history_bronze_script} "
                "--oot-start '{{ dag_run.conf.get(\"oot_start\", \"2025-01-01\") }}' "
                "--oot-end   '{{ dag_run.conf.get(\"oot_end\",   \"2025-03-31\") }}' "
                "--no-hist --download-data"
            ),
        )

        weather_download_and_hist >> weather_oot_range

    # ------------------------------
    # Group 2: Forecast (GFS)
    # ------------------------------
    with TaskGroup(group_id="gfs_forecast") as tg_forecast:

        # 2A) Backfill OOT forecasts for the SAME date range (per valid_date file)
        # You can trim cycles/leads for speed; these are balanced defaults.
        forecast_oot_range = BashOperator(
            task_id="forecast_oot_range",
            bash_command=(
                "set -euo pipefail && "
                f"cd {SCRIPTS_DIR} && "
                f"python3 {forecast_script} "
                "--oot-start '{{ dag_run.conf.get(\"oot_start\", \"2025-01-01\") }}' "
                "--oot-end   '{{ dag_run.conf.get(\"oot_end\",   \"2025-03-31\") }}' "
                "--cycles 0,12 --fhours 6,12,24,48,72 "
                "--out datamart/bronze/gfs_airports "
                "--no-hist"
            ),
        )

    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.ALL_DONE)

    # Orchestrate: download/write history -> OOT weather + OOT forecast in parallel -> done
    start >> tg_weather >> [tg_forecast] >> done

# ====================================================================
# DAG: monthly data pipeline (kept as-is, now includes Flight Bronze OOT)
# ====================================================================
with DAG(
    dag_id='dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    tags=['pipeline', 'monthly'],
) as dag:

    # -------------------------
    # Label Store
    # -------------------------
    dep_check_source_label_data = EmptyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'set -euo pipefail && '
            f'cd {SCRIPTS_DIR} && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = EmptyOperator(task_id="silver_label_store")
    gold_label_store   = EmptyOperator(task_id="gold_label_store")
    label_store_completed = EmptyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # -------------------------
    # Flight Bronze (OOT) — runs each schedule with ds
    # -------------------------
    with TaskGroup(group_id="flight_bronze_oot") as flight_bronze_oot:
        dep_check_flight_oot = EmptyOperator(task_id="dep_check_flight_oot")

        run_bronze_flight_oot = BashOperator(
            task_id='run_bronze_flight_oot',
            bash_command=(
                'set -euo pipefail && '
                f'cd {SCRIPTS_DIR} && '
                f'python3 {FLIGHT_BRONZE_SCRIPT} '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        flight_bronze_oot_done = EmptyOperator(task_id="flight_bronze_oot_done")
        dep_check_flight_oot >> run_bronze_flight_oot >> flight_bronze_oot_done

    # -------------------------
    # Weather History Bronze (OOT) — runs each schedule with ds
    # -------------------------
    with TaskGroup(group_id="flight_weather_oot") as flight_weather_oot:
        dep_check_weather_oot = EmptyOperator(task_id="dep_check_weather_oot")

        run_bronze_weather_infer = BashOperator(
            task_id='run_bronze_weather_infer',
            bash_command=(
                'set -euo pipefail && '
                f'cd {SCRIPTS_DIR} && '
                f'python3 {weather_history_bronze_script} '
                '--snapshotdate "{{ ds }}" --nohist --download-data'
            ),
        )

        flight_weather_oot_done = EmptyOperator(task_id="weather_bronze_oot_done")
        dep_check_weather_oot >> run_bronze_weather_infer >> flight_weather_oot_done

    # -------------------------
    # Forecast History Bronze (OOT) — runs each schedule with ds
    # -------------------------
    with TaskGroup(group_id="flight_weather_oot") as flight_forecast_oot:
        dep_check_forecast_oot = EmptyOperator(task_id="dep_check_forecast_oot")

        run_bronze_forecast_infer = BashOperator(
            task_id="run_bronze_forecast_infer",
            bash_command=(
                "set -euo pipefail && "
                f"cd {SCRIPTS_DIR} && "
                "python3 bronze_forecast_store.py "
                f"--oot-start '{{ ds }}' --oot-end '{{ ds }}' "
                "--no-hist"
            ),
        )
        flight_forecast_oot_done = EmptyOperator(task_id="forecast_bronze_oot_done")
        dep_check_forecast_oot >> run_bronze_forecast_infer >> flight_forecast_oot_done

    # -------------------------
    # Feature Store (now depends on Flight Bronze OOT)
    # -------------------------
    dep_check_source_data_bronze_1 = EmptyOperator(task_id="dep_check_source_data_bronze_1")
    dep_check_source_data_bronze_2 = EmptyOperator(task_id="dep_check_source_data_bronze_2")
    dep_check_source_data_bronze_3 = EmptyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_table_1 = EmptyOperator(task_id="bronze_table_1")
    bronze_table_2 = EmptyOperator(task_id="bronze_table_2")
    bronze_table_3 = EmptyOperator(task_id="bronze_table_3")

    silver_table_1 = EmptyOperator(task_id="silver_table_1")
    silver_table_2 = EmptyOperator(task_id="silver_table_2")

    gold_feature_store = EmptyOperator(task_id="gold_feature_store")
    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    # Make Feature Store wait for Flight Bronze OOT outputs
    flight_bronze_oot >> [dep_check_source_data_bronze_1, dep_check_source_data_bronze_2, dep_check_source_data_bronze_3]

    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    gold_feature_store >> feature_store_completed

    # -------------------------
    # Model inference
    # -------------------------
    model_inference_start = EmptyOperator(task_id="model_inference_start")
    model_1_inference = EmptyOperator(task_id="model_1_inference")
    model_2_inference = EmptyOperator(task_id="model_2_inference")
    model_inference_completed = EmptyOperator(task_id="model_inference_completed")

    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed

    # -------------------------
    # Model monitoring
    # -------------------------
    model_monitor_start = EmptyOperator(task_id="model_monitor_start")
    model_1_monitor = EmptyOperator(task_id="model_1_monitor")
    model_2_monitor = EmptyOperator(task_id="model_2_monitor")
    model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")

    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed

    # -------------------------
    # AutoML retraining
    # -------------------------
    model_automl_start = EmptyOperator(task_id="model_automl_start")
    model_1_automl = EmptyOperator(task_id="model_1_automl")
    model_2_automl = EmptyOperator(task_id="model_2_automl")
    model_automl_completed = EmptyOperator(task_id="model_automl_completed")

    [feature_store_completed, label_store_completed] >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed


# ====================================================================
# Optional DAG: one-off historical Flight Bronze (manual trigger)
#   - No schedule; runs your script WITHOUT --snapshotdate
# ====================================================================
with DAG(
    dag_id='flight_bronze_historical_once',
    default_args=default_args,
    description='Run Flight Bronze historical batch once (manual trigger)',
    schedule_interval=None,            # manual only
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['flight', 'bronze', 'historical'],
) as dag_hist:

    start = EmptyOperator(task_id="start")
    run_bronze_flight_historical = BashOperator(
        task_id='run_bronze_flight_historical',
        bash_command=(
            'set -euo pipefail && '
            f'cd {SCRIPTS_DIR} && '
            f'python3 {FLIGHT_BRONZE_SCRIPT}'
        ),
    )
    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.ALL_DONE)

    start >> run_bronze_flight_historical >> done
