# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
from typing import Dict, Any, Optional

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import log_exec_time
from lightautoml.tasks import Task

logger = logging.getLogger(__name__)


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[Dict] = None) -> Dict[str, Any]:
    with log_exec_time("LAMA"):
        # to assure that LAMA correctly interprets these columns as categorical
        roles = roles if roles else {}
        dtype = dtype if dtype else {}

        data = pd.read_csv(path,  dtype=dtype)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        task = Task(task_type)

        automl = TabularAutoML(task=task, timeout=3600 * 3, general_params={"use_algos": use_algos})

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    if metric_name == "mse":
        evaluator = sklearn.metrics.mean_squared_error
    elif metric_name == "areaUnderROC":
        evaluator = sklearn.metrics.roc_auc_score
    else:
        raise ValueError(f"Metric {metric_name} is not supported")

    metric_value = evaluator(train_data[target_col].values, oof_predictions.data[:, 0])
    logger.info(f"mse score for out-of-fold predictions: {metric_value}")

    with log_exec_time():
        te_pred = automl.predict(test_data)

    test_metric_value = evaluator(test_data[target_col].values, te_pred.data[:, 0])
    logger.info(f"mse score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    return {"metric_value": metric_value, "test_metric_value": test_metric_value}
