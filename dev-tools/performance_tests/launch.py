import logging
import logging.config
from copy import deepcopy
from pprint import pprint
from typing import Any, Callable
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
    config = {
        "path": "examples/data/small_used_cars_data.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": USED_CARS_DATASET_ROLES["target"],
        "use_algos": ["lgb"],
        "roles": USED_CARS_DATASET_ROLES,
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str'
        }
    }

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
    calculate_quality(lama_automl)
    # calculate_quality(spark_automl)
