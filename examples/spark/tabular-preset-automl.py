import logging.config
import os
from typing import Tuple

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    data.unpersist()

    return train_data, test_data


def get_spark_session():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:
        spark_sess = (
            SparkSession
            .builder
            .master("local[*]")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config("spark.cleaner.referenceTracking", "true")
            .config("spark.cleaner.periodicGC.interval", "1min")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "12g")
            .config("spark.executor.memory", "12g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 5
    use_algos = [["lgb", "linear_l2"], ["lgb"]]

    path = "file:///opt/spark_data/small_used_cars_data_cleaned.csv"
    task_type = "reg"
    roles = {
        "target": "price",
        "drop": ["dealer_zip", "description", "listed_date",
                 "year", 'Unnamed: 0', '_c0',
                 'sp_id', 'sp_name', 'trimId',
                 'trim_name', 'major_options', 'main_picture_url',
                 'interior_color', 'exterior_color'],
        "numeric": ['latitude', 'longitude', 'mileage']
    }

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)
        train_data, test_data = prepare_test_and_train(spark, path, seed)

        test_data_dropped = test_data

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            lgb_params={'use_single_dataset_mode': True},
            linear_l2_params={"default_params": {"regParam": [1]}},
            reader_params={"cv": cv, "advanced_roles": False}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    transformer = automl.make_transformer()

    # we delete this variable to make garbage collection of a local checkpoint
    # used to produce Spark DataFrame with predictions possible
    del oof_predictions
    automl.release_cache()

    with log_exec_timer("saving model") as saving_timer:
        transformer.write().overwrite().save("hdfs://namenode:9000/automl_pipeline")

    with log_exec_timer("spark-lama predicting on test (#1 way)") as predict_timer:
        te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

        logger.info(f"score for test predictions: {test_metric_value}")

    with log_exec_timer("spark-lama predicting on test (#2 way)") as predict_timer_2:
        te_pred = automl.make_transformer().transform(test_data_dropped)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        test_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))

        logger.info(f"score for test predictions: {test_metric_value}")

        expected_predictions_sum = te_pred.select(F.sum(pred_column).alias("sum")).collect()[0]["sum"]
        logger.info(f"expected predictions sum: {expected_predictions_sum}")

    with log_exec_timer("Loading model time") as loading_timer:
        pipeline_model = PipelineModel.load("hdfs://namenode:9000/automl_pipeline")

    with log_exec_timer("spark-lama predicting on test (#3 way)") as predict_timer_3:
        te_pred = pipeline_model.transform(test_data_dropped)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        test_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))

        logger.info(f"score for test predictions via loaded pipeline: {test_metric_value}")

        actual_predictions_sum = te_pred.select(F.sum(pred_column).alias("sum")).collect()[0]["sum"]
        logger.info(f"actual predictions sum: {actual_predictions_sum}")

    logger.info("Predicting is finished")

    result = {
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration,
        "saving_duration_secs": saving_timer.duration,
        "loading_duration_secs": loading_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    train_data.unpersist()
    test_data.unpersist()

    spark.stop()
