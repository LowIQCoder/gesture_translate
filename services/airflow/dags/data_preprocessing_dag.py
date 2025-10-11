from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

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
    pass
