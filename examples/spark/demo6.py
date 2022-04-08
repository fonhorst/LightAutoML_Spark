"""
Spark AutoML with nested CV usage
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

    use_algos = [["linear_l2", "lgb"], ["linear_l2", "lgb"]]


    with log_exec_timer("spark-lama training and predict") as train_timer:

        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            timeout=1800,
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 3000}},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={
                "use_algos": use_algos,
                "nested_cv": True,
                "skip_conn": True,
            },
            nested_cv_params={"cv": 5, "n_folds": None},
            reader_params={"advanced_roles": False}
        )

        oof_pred = automl.fit_predict(train, roles=roles)
        score = task.get_dataset_metric()
        metric_value = score(oof_pred)
        logger.info(f"Score for out-of-fold predictions: {metric_value}")

        test_pred = automl.predict(test, add_reader_attrs=True)
        score = task.get_dataset_metric()
        metric_value = score(test_pred)
        logger.info(f"Score for test predictions: {metric_value}")
