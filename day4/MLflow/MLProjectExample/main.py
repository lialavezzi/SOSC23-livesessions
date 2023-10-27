import mlflow
import click
import os
import sys


# this is our pipeline

@click.command()
@click.option("--data-url", default="tt", type=str)
def workflow(data_url):
    # 1st entrypoint
    download_data_run = mlflow.run(".", "download_data", parameters={"data_url": data_url}, env_manager="local") 
    download_data_run.wait()
    artifact_path = mlflow.get_run(download_data_run.run_id).info.artifact_uri
    # 2nd entrypoint
    prepare_train_test_run = mlflow.run(".", "prepare_train_test", parameters={"csv_path": "/".join([artifact_path, 'dataset.csv'])}, env_manager="local") 
    prepare_train_test_run.wait()
    artifact_path = mlflow.get_run(prepare_train_test_run.run_id).info.artifact_uri
    # 3rd entrypoint
    train_run = mlflow.run(".", "train", parameters={"csv_train_path": "/".join([artifact_path, 'train.csv']), "csv_test_path": "/".join([artifact_path, 'test.csv'])}, env_manager="local") 

if __name__ == "__main__":
    workflow()