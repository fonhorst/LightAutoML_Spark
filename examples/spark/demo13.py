"""
Spark AutoML with valid_data passed in fit_predict
"""
import logging.config
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config
logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    spark = get_spark_session()

    dataset_path, task_type, roles, dtype = get_dataset_attrs("lama_test_dataset")

    train, test = prepare_test_and_train(spark, dataset_path, seed=42)

    use_algos = [["linear_l2", "lgb"]]

    with log_exec_timer("spark-lama training and predict") as train_timer:

        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            timeout=180000,
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 3000}},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"advanced_roles": False}
        )

        oof_pred = automl.fit_predict(train, roles=roles, valid_data=test)
        score = task.get_dataset_metric()
        oof_metric_value = score(oof_pred)

        test_pred = automl.predict(test, add_reader_attrs=True)
        score = task.get_dataset_metric()
        metric_value = score(test_pred)

        logger.info(f"Score for out-of-fold predictions: {oof_metric_value}")
        logger.info(f"Score for test predictions: {metric_value}")
