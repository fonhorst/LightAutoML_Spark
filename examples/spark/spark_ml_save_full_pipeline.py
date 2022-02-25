import logging.config

from typing import Tuple

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config
from lightautoml.spark.utils import spark_session


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def prepare_test_and_train(spark: SparkSession, path: str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
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


if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:
        roles = {
            "target": 'price',
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            "numeric": ['latitude', 'longitude', 'mileage']
        }

        task = SparkTask("reg")
        train_data, test_data = prepare_test_and_train(spark, "examples/data/tiny_used_cars_data.csv", seed=42)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": ["linear_l2"]},
            reader_params={"cv": 3},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
        )

        _ = automl.fit_predict(train_data, roles)

        with log_exec_timer("spark-lama predicting on test") as predict_timer:
            pred = automl.predict(test_data, add_reader_attrs=True)

        transformer = automl.make_transformer()
        transformer.write().overwrite().save("/tmp/full_pipeline")

        pred = transformer.transform(train_data) # test_data
        pred.toPandas().to_csv("/tmp/full_pipeline_prediction.csv")

        pipeline_model = PipelineModel.load("/tmp/full_pipeline")
        pred = pipeline_model.transform(train_data) # test_data
        pred.toPandas().to_csv("/tmp/full_pipeline_prediction_loaded_pipe.csv")

        logger.info("Finished")
