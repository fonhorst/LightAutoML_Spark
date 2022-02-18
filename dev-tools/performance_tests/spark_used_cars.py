# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
from typing import Dict, Any, Optional

import sklearn
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import log_exec_time, spark_session
from lightautoml.utils.tmp_utils import log_data

logger = logging.getLogger(__name__)


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     cv: int = 5,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[None] = None) -> Dict[str, Any]:
    roles = roles if roles else {}

    with spark_session(master="local[4]") as spark:
        with log_exec_time("spark-lama training"):
            task = SparkTask(task_type)
            data = spark.read.csv(path, header=True, escape="\"").repartition(4)

            data = (
                data
                # .withColumnRenamed(target_col, f"{target_col}_old")
                # .select('*', F.col(f"{target_col}_old").astype(DoubleType()).alias(target_col)).drop(f"{target_col}_old")
                .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
                .withColumn('is_test', F.rand(seed))
                .cache()
            )
            data.write.mode('overwrite').format('noop').save()
            # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

            train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
            test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

            train_data.write.mode('overwrite').format('noop').save()
            test_data.write.mode('overwrite').format('noop').save()

            # test_data_dropped = test_data \
            #     .drop(F.col(target_col))
            test_data_dropped = test_data

            automl = SparkTabularAutoML(
                spark=spark,
                task=task,
                general_params={"use_algos": use_algos},
                reader_params={"cv": cv}
            )

            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles
            )

        log_data("spark_test_part", {"test": test_data.select(SparkDataset.ID_COLUMN, target_col).toPandas()})

        logger.info("Predicting on out of fold")

        # predict_col = oof_predictions.features[0]
        # oof_preds_for_eval = (
        #     oof_predictions.data
        #     .join(train_data, on=SparkDataset.ID_COLUMN)
        #     .select(SparkDataset.ID_COLUMN, target_col, predict_col)
        # )
        #
        # if metric_name == "mse":
        #     evaluator = sklearn.metrics.mean_squared_error
        # elif metric_name == "areaUnderROC":
        #     evaluator = sklearn.metrics.roc_auc_score
        # else:
        #     raise ValueError(f"Metric {metric_name} is not supported")
        #
        # oof_preds_for_eval_pdf = oof_preds_for_eval.toPandas()
        # metric_value = evaluator(oof_preds_for_eval_pdf[target_col].values, oof_preds_for_eval_pdf[predict_col].values)

        score = task.get_dataset_metric()
        metric_value = score(oof_predictions)

        logger.info(f"{metric_name} score for out-of-fold predictions: {metric_value}")

        with log_exec_time("spark-lama predicting on test"):
            te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

            # te_pred = (
            #     te_pred.data
            #     .join(test_data, on=SparkDataset.ID_COLUMN)
            #     .select(SparkDataset.ID_COLUMN, target_col, te_pred.features[0])
            # )
            #
            # # test_metric_value = evaluator.evaluate(te_pred)
            # # logger.info(f"{evaluator.getMetricName()} score for test predictions: {test_metric_value}")
            #
            # te_pred_pdf = te_pred.toPandas()
            # test_metric_value = evaluator(te_pred_pdf[target_col].values, te_pred_pdf[predict_col].values)

            score = task.get_dataset_metric()
            test_metric_value = score(te_pred)

            # alternative way of measuring (gives the same results)
            # te_pred_df = te_pred.data.join(
            #     test_data.select(SparkDataset.ID_COLUMN, F.col(target_col).astype(FloatType()).alias(target_col)),
            #     on=SparkDataset.ID_COLUMN
            # )
            # ds = SparkDataset(te_pred_df, te_pred.roles, te_pred.task, target=target_col)
            # test_metric_value = score(ds)

            logger.info(f"{metric_name} score for test predictions: {test_metric_value}")

        logger.info("Predicting is finished")

        return {"metric_value": metric_value, "test_metric_value": test_metric_value}
