from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello():
    print("Airflow talking to Postgres metadata DB just fine!")

with DAG("smoke_db",
         start_date=datetime(2025, 1, 1),
         schedule=None, catchup=False) as dag:
    PythonOperator(task_id="hello", python_callable=hello)
