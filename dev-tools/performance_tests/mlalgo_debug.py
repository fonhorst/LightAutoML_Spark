import pickle

import sklearn

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.validation.base import DummyIterator

import numpy as np

# TODO: need to log data in predict
# TODO: correct order in PandasDataset from Spark ?
# TODO: correct parameters of BoostLGBM?
# TODO: correct parametes of Tuner for BoostLGBM?

mode = "spark"

path = f'../datalog_{mode}_lgb_train_val.pickle'
path_for_test = f'../datalog_{mode}_test_part.pickle'
path_predict = f'../datalog_{mode}_lgb_predict.pickle'
target_col = 'TARGET'

with open(path, "rb") as f:
    data = pickle.load(f)
    train = data['data']['train']
    valid = data['data']['valid']

with open(path_for_test, "rb") as f:
    test_target_df = pickle.load(f)
    test_target_df = test_target_df['data']['test']

#     # if mode == "spark":
#     #     test_target_df.sort_values(SparkDataset.ID_COLUMN, inplace=True)

with open(path_predict, "rb") as f:
    test_df = pickle.load(f)
    test_df = test_df['data']['predict']

# TODO: verification by _id equality

tgts = test_df.target

if mode == "spark":
    tgts = test_df.target
else:
    tgts = test_target_df[target_col].values

train_valid = DummyIterator(train)
ml_algo = BoostLGBM()

ml_algo, _ = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)

preds = ml_algo.predict(test_df)

evaluator = sklearn.metrics.roc_auc_score

test_metric_value = evaluator(tgts, preds.data[:, 0])

print(f"Test metric value: {test_metric_value}")
