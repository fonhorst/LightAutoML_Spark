import logging
import logging.config

from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT
import pandas as pd

from lightautoml.tasks import Task


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    seed = 42
    cv = 5
    # use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb"]]

    path = "/opt/spark_data/small_used_cars_data_cleaned.csv"
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
    dtype = {
        'fleet': 'str', 'frame_damaged': 'str',
        'has_accidents': 'str', 'isCab': 'str',
        'is_cpo': 'str', 'is_new': 'str',
        'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
    }

    with log_exec_timer("LAMA") as train_timer:
        # to assure that LAMA correctly interprets these columns as categorical
        dtype = dtype if dtype else {}

        data = pd.read_csv(path,  dtype=dtype)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        task = Task(task_type)

        automl = TabularAutoML(
            task=task,
            timeout=3600 * 3,
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

    logger.info(f"Score for out-of-fold predictions: {metric_value}")

    with log_exec_timer() as predict_timer:
        te_pred = automl.predict(test_data)
        te_pred.target = test_data[roles['target']]

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

    logger.info(f"mse score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    result = {
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration
    }

    print(f"EXP-RESULT: {result}")
