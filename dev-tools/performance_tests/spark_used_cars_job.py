# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
import logging.config
import yaml
import json
import os
import socket
import sys
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import DoubleType

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logger = logging.getLogger(__name__)


def calculate_automl(spark: SparkSession, path: str, seed: int = 42, use_algos=("lgb", "linear_l2")) -> Dict[str, Any]:
    with log_exec_timer() as fit_predict_time:
        target_col = "price"
        task = SparkTask("reg")
        data = spark.read.csv(path, header=True, escape='"')

        data = (
            data.withColumnRenamed(target_col, f"{target_col}_old")
            .select("*", F.col(f"{target_col}_old").astype(DoubleType()).alias(target_col))
            .drop(f"{target_col}_old")
            .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
            .cache()
        )
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

        test_data_dropped = test_data.drop(F.col(target_col)).cache()

        automl = TabularAutoML(spark=spark, task=task, general_params={"use_algos": use_algos})

        oof_predictions = automl.fit_predict(
            train_data,
            roles={
                "target": target_col,
                "drop": [
                    "dealer_zip",
                    "description",
                    "listed_date",
                    "year",
                    "Unnamed: 0",
                    "_c0",
                    "sp_id",
                    "sp_name",
                    "trimId",
                    "trim_name",
                    "major_options",
                    "main_picture_url",
                    "interior_color",
                    "exterior_color",
                ],
                "numeric": ["latitude", "longitude", "mileage"],
            },
        )

    logger.info("Predicting on out of fold")

    oof_preds_for_eval = oof_predictions.data.join(train_data, on=SparkDataset.ID_COLUMN).select(
        SparkDataset.ID_COLUMN, target_col, oof_predictions.features[0]
    )

    evaluator = RegressionEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col, metricName="mse")

    metric_value = evaluator.evaluate(oof_preds_for_eval)
    logger.info(f"{evaluator.getMetricName()} score for out-of-fold predictions: {metric_value}")

    # TODO: SPARK-LAMA fix bug in SparkToSparkReader.read method
    with log_exec_timer() as predict_time:
        te_pred = automl.predict(test_data_dropped)

        te_pred = te_pred.data.join(test_data, on=SparkDataset.ID_COLUMN).select(
            SparkDataset.ID_COLUMN, target_col, te_pred.features[0]
        )

        test_metric_value = evaluator.evaluate(te_pred)
        logger.info(f"{evaluator.getMetricName()} score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    return {
        "fit_predict_time": fit_predict_time.t,
        "predict_time": predict_time.t,
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "completion_datetime": f"{datetime.now()}",
    }


@contextmanager
def configure_spark_session(do_configuring: bool, name: str, config_data: dict):
    if do_configuring:
        local_ip = socket.gethostbyname(socket.gethostname())
        spark_sess_builder = (
            SparkSession.builder.appName(name)
            .master("k8s://https://node2.bdcl:6443")
            .config("spark.driver.host", local_ip)
            .config("spark.driver.bindAddress", "0.0.0.0")
        )

        for arg, value in config_data.items():
            spark_sess_builder = spark_sess_builder.config(arg, value)

        spark = spark_sess_builder.getOrCreate()

    else:
        spark = SparkSession.builder.getOrCreate()

    try:
        yield spark
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.INFO, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
    logger = logging.getLogger(__name__)

    do_configuring = True if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else False

    # Read values from config file
    with open("/scripts/config.yaml", "r") as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Delete non-spark parameters
    app_name = config_data["name"]
    dataset_path = config_data["dataset_path"]
    use_state_file = config_data["use_state_file"]
    del config_data["name"]
    del config_data["dataset_path"]
    del config_data["use_state_file"]

    # Launch jobs with experiments and write results into file
    with configure_spark_session(do_configuring, app_name, config_data) as spark:
        res = calculate_automl(spark, path=f"/spark_data/{dataset_path}", use_algos=["lgb", "linear_l2"])
        res_dict = {}
        res_dict[app_name] = res

        # Write in experiment log history file
        with open(f"/exp_results/{app_name}.log", "a+") as file:
            for key, val in res_dict.items():
                file.write(f"{key}:{val}\n")
            file.write("-----------\n")

        print(f"Test results:[{res}]")
