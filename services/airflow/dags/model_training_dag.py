from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

with DAG(
    dag_id="data_preprocessing",
    description="Dag for aquiring and preprocessing data",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5)
    },
    schedule=timedelta(days=1),
    tags=["datadads"]
) as dag:
    
    # Preprocess task
    preprocess_task = BashOperator(
        task_id="preprocess_task",
        bash_command=f"cd {PROJECT_DIR} && python3 -m src.data.processing"
    )

    # Train taskk
    train_task = BashOperator(
        task_id="train_task",
        bash_command=f"cd {PROJECT_DIR} && python3 -m src.ml.train"
    )

    preprocess_task >> train_task
