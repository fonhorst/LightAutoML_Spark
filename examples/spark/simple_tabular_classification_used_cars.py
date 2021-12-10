# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import time

import logging
import sys

formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s {%(module)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)
# set up logging to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console)

from contextlib import contextmanager

import pandas as pd
from pyspark.sql import SparkSession

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# load and prepare data
# TODO: put a correct path for used_cars dataset
from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import print_exec_time

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('lightautoml')]
for logger in loggers:
    logger.setLevel(logging.INFO)
    # logger.addHandler(console)

data = pd.read_csv("examples/data/tiny_used_cars_data.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


@contextmanager
def spark_session(parallelism: int = 1) -> SparkSession:
    spark = (
        SparkSession
        .builder
        .appName("SPARK-LAMA-app")
        .master("spark://node4.bdcl:7077")
        # .master("local[4]")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.driver.host", "node4.bdcl")
        .config("spark.driver.cores", "4")
        .config("spark.driver.memory", "16g")
        .config("spark.cores.max", "12")
        .config("spark.executor.instances", "3")
        .config("spark.executor.memory", "16g")
        .config("spark.executor.cores", "4")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    # time.sleep(600)
    try:
        yield spark
    finally:
        # time.sleep(600)
        spark.stop()


# run automl
if __name__ == "__main__":
    with spark_session(parallelism=4) as spark:
        task = SparkTask("reg")

        automl = TabularAutoML(
            spark=spark,
            task=task,
            # general_params={"use_algos": ["lgb", "linear_l2"]}
            # general_params={"use_algos": ["linear_l2"]}
            general_params={"use_algos": ["lgb"]}
        )

        with print_exec_time():
            oof_predictions = automl.fit_predict(
                train_data,
                roles={"target": "price", "drop": ["dealer_zip", "description", "listed_date"]}
            )

        # TODO: SPARK-LAMA fix bug in SparkToSparkReader with nans processing to make it working on test data
        # te_pred = automl.predict(test_data)

        # # calculate scores
        # # TODO: replace with mse
        # #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
        # print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        # print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
