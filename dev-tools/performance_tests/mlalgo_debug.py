import pickle

import sklearn

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.validation.base import DummyIterator

# TODO: need to log data in predict
# TODO: correct order in PandasDataset from Spark ?
# TODO: correct parameters of BoostLGBM?
# TODO: correct parametes of Tuner for BoostLGBM?

path = './datalog_lama_lgb_train_val.pickle'
path_for_test = './datalog_lama_test_part.pickle'
path_predict = './datalog_lama_lgb_predict.pickle'
target_col = 'TARGET'

with open(path, "rb") as f:
    data = pickle.load(f)
    train = data['train']
    valid = data['valid']

with open(path_for_test, "rb") as f:
    test_target_df = pickle.load(f)
    test_target_df = test_target_df['test']

with open(path_predict, "rb") as f:
    test_df = pickle.load(f)
    test_df = test_df['predict']

train_valid = DummyIterator(train)
ml_algo = BoostLGBM()

ml_algo, _ = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)

preds = ml_algo.predict(test_df)

evaluator = sklearn.metrics.roc_auc_score

test_metric_value = evaluator(test_target_df[target_col].values, preds.data[:, 0])