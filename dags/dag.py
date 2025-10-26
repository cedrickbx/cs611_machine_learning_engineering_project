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