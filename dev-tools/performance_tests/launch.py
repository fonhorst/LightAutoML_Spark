import logging
import logging.config
from copy import deepcopy
from pprint import pprint
from typing import Any, Callable

from pyspark.ml import Pipeline

from lama_used_cars import calculate_automl as lama_automl
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from spark_used_cars import calculate_automl as spark_automl


USED_CARS_DATASET_ROLES = {
    "target": "price",
    "drop": ["dealer_zip", "description", "listed_date",
             "year", 'Unnamed: 0', '_c0',
             'sp_id', 'sp_name', 'trimId',
             'trim_name', 'major_options', 'main_picture_url',
             'interior_color', 'exterior_color'],
    "numeric": ['latitude', 'longitude', 'mileage']
}


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def calculate_quality(calc_automl: Callable):
    # used_cars dataset
    config = {
        "path": "examples/data/small_used_cars_data.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": USED_CARS_DATASET_ROLES["target"],
        # "use_algos": [["lgb", "linear_l2"], ["lgb", "linear_l2"]],
        "use_algos": ["linear_l2"],
        "roles": USED_CARS_DATASET_ROLES,
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str'
        }
    }

    # #  LAMA's test set
    # config = {
    #     "path": "./examples/data/sampled_app_train.csv",
    #     "task_type": "binary",
    #     "metric_name": "areaUnderROC",
    #     "target_col": "TARGET",
    #     "use_algos": ["lgb"],
    #     "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    # }

    # # https://www.openml.org/d/734
    # config = {
    #     "path": "/opt/ailerons.csv",
    #     "task_type": "binary",
    #     "metric_name": "areaUnderROC",
    #     "target_col": "binaryClass",
    #     "use_algos": ["linear_l2"],
    #     "roles": {"target": "binaryClass"},
    # }

    # # https://www.openml.org/d/4534
    # config = {
    #     "path": "/opt/PhishingWebsites.csv",
    #     "task_type": "binary",
    #     "metric_name": "areaUnderROC",
    #     "target_col": "Result",
    #     "use_algos": ["lgb"],
    #     "roles": {"target": "Result"},
    # }

    # # https://www.openml.org/d/981
    # config = {
    #     "path": "/opt/kdd_internet_usage.csv",
    #     "task_type": "binary",
    #     "metric_name": "areaUnderROC",
    #     "target_col": "Who_Pays_for_Access_Work",
    #     "use_algos": ["lgb"],
    #     "roles": {"target": "Who_Pays_for_Access_Work"},
    # }

    # # https://www.openml.org/d/42821
    # config = {
    #     "path": "/opt/nasa_phm2008.csv",
    #     "task_type": "reg",
    #     "metric_name": "mse",
    #     "target_col": "class",
    #     "use_algos": ["lgb"],
    #     "roles": {"target": "class"},
    # }

    # # https://www.openml.org/d/4549
    # config = {
    #     "path": "/opt/Buzzinsocialmedia_Twitter_25k.csv",
    #     "task_type": "reg",
    #     "metric_name": "mse",
    #     "target_col": "Annotation",
    #     "use_algos": ["lgb"],
    #     "roles": {"target": "Annotation"},
    # }

    # seeds = [1, 42, 100, 200, 333, 555, 777, 2000, 50000, 100500,
    #              200000, 300000, 1_000_000, 2_000_000, 5_000_000, 74909, 54179, 68572, 25425]

    seeds = [42]
    results = []
    for seed in seeds:
        cfg = deepcopy(config)
        cfg['seed'] = seed
        res = calc_automl(**cfg)
        results.append(res)
        logger.info(f"Result for seed {seed}: {res}")

    mvals = [f"{r['metric_value']:_.2f}" for r in results]
    print("OOf on train metric")
    pprint(mvals)

    test_mvals = [f"{r['test_metric_value']:_.2f}" for r in results]
    print("Test metric")
    pprint(test_mvals)


if __name__ == "__main__":
    # calculate_quality(lama_automl)
    calculate_quality(spark_automl)
