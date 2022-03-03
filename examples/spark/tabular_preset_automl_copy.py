import logging.config
from typing import Tuple

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, spark_session, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")  # .repartition(4)

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()
    # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


spark = SparkSession.builder.getOrCreate()

seed = 42
cv = 5
use_algos = [["lgb", "linear_l2"], ["lgb"]]
path = "/opt/spark_data/small_used_cars_data.csv"
task_type = "reg"
roles = {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            }

with log_exec_timer("spark-lama training") as train_timer:
    task = SparkTask(task_type)
    train_data, test_data = prepare_test_and_train(spark, path, seed)

    test_data_dropped = test_data

    automl = SparkTabularAutoML(
        spark=spark,
        task=task,
        general_params={"use_algos": use_algos},
        reader_params={"cv": cv, "advanced_roles": False},
        tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
    )

    oof_predictions = automl.fit_predict(
        train_data,
        roles=roles
    )

logger.info("Predicting on out of fold")

score = task.get_dataset_metric()
metric_value = score(oof_predictions)

logger.info(f"score for out-of-fold predictions: {metric_value}")

with log_exec_timer("spark-lama predicting on test") as predict_timer:
    te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

    score = task.get_dataset_metric()
    test_metric_value = score(te_pred)

    logger.info(f"score for test predictions: {test_metric_value}")

logger.info("Predicting is finished")

result = {
    "metric_value": metric_value,
    "test_metric_value": test_metric_value,
    "train_duration_secs": train_timer.duration,
    "predict_duration_secs": predict_timer.duration
}

print(f"EXP-RESULT: {result}")

spark.stop()