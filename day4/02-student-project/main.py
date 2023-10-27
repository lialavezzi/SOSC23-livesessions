import mlflow
import click
import os
import sys
import warnings

warnings.filterwarnings('ignore')

@click.command()
## This is an example of how you can pass parameters at MFlow runtime
@click.option("--data-path", default="tt", type=str)
def workflow(data_path):

    # 1st entrypoint
    train = mlflow.run(".", "train", env_manager="local") #parameters={"data_path": data_path}, env_manager="local") 
    train.wait()
    model_run_uri = mlflow.get_run(train.run_id).info.artifact_uri
    # 2nd entrypoint
    evaluate = mlflow.run(".", "evaluate", parameters={"model_run_uri": "/".join([model_run_uri, 'classifier.keras'])}, env_manager="local") 
    evaluate.wait()

if __name__ == "__main__":
    workflow()