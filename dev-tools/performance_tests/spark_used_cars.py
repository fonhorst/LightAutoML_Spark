# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
import logging.config
import sys
from contextlib import contextmanager
from typing import Dict, Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import DoubleType

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import log_exec_time, spark_session, logging_config, VERBOSE_LOGGING_FORMAT

logger = logging.getLogger(__name__)


def calculate_automl(spark: SparkSession, path:str, seed: int = 42, use_algos = ("lgb", "linear_l2")) -> Dict[str, Any]:
    with log_exec_time():
        target_col = 'price'
        task = SparkTask("reg")
        data = spark.read.csv(path, header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/tiny_used_cars_data.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/0125x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/4x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/2x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///opt/0125l_dataset.csv", header=True, escape="\"")
        data = data.withColumnRenamed(target_col, f"{target_col}_old") \
            .select('*', F.col(f"{target_col}_old").astype(DoubleType()).alias(target_col)).drop(f"{target_col}_old") \
            .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id()) \
            .cache()
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

        test_data_dropped = test_data \
            .drop(F.col(target_col)).cache()

        automl = TabularAutoML(spark=spark, task=task, general_params={"use_algos": use_algos})

        oof_predictions = automl.fit_predict(
            train_data,
            roles={
                "target": target_col,
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                "numeric": ['latitude', 'longitude', 'mileage']
            }
        )

    logger.info("Predicting on out of fold")

    oof_preds_for_eval = (
        oof_predictions.data
        .join(train_data, on=SparkDataset.ID_COLUMN)
        .select(SparkDataset.ID_COLUMN, target_col, oof_predictions.features[0])
    )

    evaluator = RegressionEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
                                    metricName="mse")

    metric_value = evaluator.evaluate(oof_preds_for_eval)
    logger.info(f"{evaluator.getMetricName()} score for out-of-fold predictions: {metric_value}")

    # TODO: SPARK-LAMA fix bug in SparkToSparkReader.read method
    with log_exec_time():
        te_pred = automl.predict(test_data_dropped)

        te_pred = (
            te_pred.data
            .join(test_data, on=SparkDataset.ID_COLUMN)
            .select(SparkDataset.ID_COLUMN, target_col, te_pred.features[0])
        )

        test_metric_value = evaluator.evaluate(te_pred)
        logger.info(f"{evaluator.getMetricName()} score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    return {"metric_value": metric_value, "test_metric_value": test_metric_value}


@contextmanager
def configure_spark_session(do_configuring: bool):
    if do_configuring:
        spark = (
            SparkSession
            .builder
            .master("k8s://https://node2.bdcl:6443")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config('spark.kubernetes.container.image', 'node2.bdcl:5000/spark-lama-k8s:3.9-3.2.0')
            .config('spark.kubernetes.container.image.pullPolicy', 'Always')
            .config('spark.kubernetes.namespace', 'lama-exps')
            .config('spark.kubernetes.authenticate.driver.serviceAccountName', 'default')
            .config('spark.kubernetes.memoryOverheadFactor', '0.1')
            .config('spark.kubernetes.driver.label.appname', 'driver-test-submit-run')
            .config('spark.kubernetes.executor.label.appname', 'executor-test-submit-run')
            .config('spark.kubernetes.executor.deleteOnTermination', 'true')
            .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName', 'spark-lama-data')
            .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass', 'local-hdd')
            .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path', '/spark_data')
            .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly', 'true')
            .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName', 'spark-lama-data')
            .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass', 'local-hdd')
            .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path', '/spark_data')
            .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly', 'true')
            .config('spark.driver.cores', '4')
            .config('spark.driver.memory', '16g')
            .config('spark.executor.instances', '4')
            .config('spark.executor.cores', '4')
            .config('spark.executor.memory', '16g')
            .config('spark.cores.max', '16')
            .config('spark.memory.fraction', '0.6')
            .config('spark.memory.storageFraction', '0.5')
            .config('spark.sql.autoBroadcastJoinThreshold', '100MB')
            .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.getOrCreate()

    try:
        yield spark
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
    logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
    logger = logging.getLogger(__name__)

    do_configuring = True if len(sys.argv) > 1 and sys.argv[1] == '1' else False

    with configure_spark_session(do_configuring) as spark:
        calculate_automl(
            spark,
            path="file:///spark_data/tiny_used_cars_data.csv",
            use_algos=["linear_l2"]
        )
