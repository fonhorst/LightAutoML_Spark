"""
AutoML with valid_data passed in fit_predict
"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task


if __name__ == "__main__":

    np.random.seed(42)

    data = pd.read_csv("/opt/spark_data/sampled_app_train.csv")

    train, test = train_test_split(data, test_size=2000, random_state=42)

    roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}

    task = Task("binary")

    automl = TabularAutoML(
        task=task,
        timeout=1800,
        general_params={
            "use_algos": [["linear_l2", "lgb"]]
        }
    )

    # It returns preds size equivalent to valid_data size
    preds = automl.fit_predict(train, roles=roles, valid_data=test)
    # test_pred = automl.predict(test)

    # print(f"OOF score: {roc_auc_score(train[roles['target']].values, oof_pred.data)}")
    print(f"Score: {roc_auc_score(test[roles['target']].values, preds.data)}")
