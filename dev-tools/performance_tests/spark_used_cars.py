# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
from typing import Dict, Any, Optional

from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import log_exec_time, spark_session

logger = logging.getLogger(__name__)


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[None] = None) -> Dict[str, Any]:
    roles = roles if roles else {}

    with spark_session(master="local[4]") as spark:
        task = SparkTask(task_type)
        data = spark.read.csv(path, header=True, escape="\"")
        data = data.withColumnRenamed(target_col, f"{target_col}_old") \
            .select('*', F.col(f"{target_col}_old").astype(DoubleType()).alias(target_col)).drop(f"{target_col}_old") \
            .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id()) \
            .cache()
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

        test_data_dropped = test_data \
            .drop(F.col(target_col)).cache()

        automl = TabularAutoML(spark=spark, task=task, general_params={"use_algos": use_algos})

        with log_exec_time("spark-lama training"):
            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles
            )

        logger.info("Predicting on out of fold")

        oof_preds_for_eval = (
            oof_predictions.data
            .join(train_data, on=SparkDataset.ID_COLUMN)
            .select(SparkDataset.ID_COLUMN, target_col, oof_predictions.features[0])
        )

        if task_type == "reg":
            evaluator = RegressionEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
                                            metricName=metric_name)
        elif task_type == "binary":
            evaluator = BinaryClassificationEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
                                                      metricName=metric_name)
        elif task_type == "multiclass":
            evaluator = MulticlassClassificationEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
                                                          metricName=metric_name)
        else:
            raise ValueError(f"Task type {task_type} is not supported")

        metric_value = evaluator.evaluate(oof_preds_for_eval)
        logger.info(f"{evaluator.getMetricName()} score for out-of-fold predictions: {metric_value}")

        with log_exec_time("spark-lama predicting on test"):
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
