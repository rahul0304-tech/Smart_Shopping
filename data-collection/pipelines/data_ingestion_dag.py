from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# Import custom operators and helpers (to be implemented)
from data_quality import check_data_quality
from data_processors import (
    process_user_interactions,
    process_product_updates,
    update_user_segments,
    update_product_metrics
)

# DAG default arguments
default_args = {
    'owner': 'smart_shopping',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1)
}

# Create the DAG
dag = DAG(
    'smart_shopping_data_ingestion',
    default_args=default_args,
    description='Data ingestion pipeline for Smart Shopping system',
    schedule_interval=timedelta(hours=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['smart_shopping', 'data_ingestion']
)

# Start operator
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# Process user interactions
process_interactions = PythonOperator(
    task_id='process_user_interactions',
    python_callable=process_user_interactions,
    dag=dag
)

# Process product updates
process_products = PythonOperator(
    task_id='process_product_updates',
    python_callable=process_product_updates,
    dag=dag
)

# Update user segments
update_segments = PythonOperator(
    task_id='update_user_segments',
    python_callable=update_user_segments,
    dag=dag
)

# Update product metrics
update_metrics = PythonOperator(
    task_id='update_product_metrics',
    python_callable=update_product_metrics,
    dag=dag
)

# Data quality checks
check_quality = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

# End operator
end_pipeline = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Define task dependencies
start_pipeline >> [process_interactions, process_products]
process_interactions >> update_segments
process_products >> update_metrics
[update_segments, update_metrics] >> check_quality >> end_pipeline