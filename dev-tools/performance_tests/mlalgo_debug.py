import logging
import logging.config
import pickle

import sklearn
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import BooleanType
from synapse.ml.lightgbm import LightGBMClassifier

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import spark_session, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.validation.base import DummyIterator
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM as SparkBoostLGBM

from pyspark.sql import functions as F

import numpy as np

# TODO: need to log data in predict
# TODO: correct order in PandasDataset from Spark ?
# TODO: correct parameters of BoostLGBM?
# TODO: correct parametes of Tuner for BoostLGBM?
from tests.spark.unit import from_pandas_to_spark

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

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
#
evaluator = sklearn.metrics.roc_auc_score

test_metric_value = evaluator(tgts, preds.data[:, 0])

print(f"Test metric value: {test_metric_value}")

# =================================================

with spark_session('local[4]') as spark:
    train_sds = from_pandas_to_spark(train, spark, train.target)
    test_sds = from_pandas_to_spark(test_df, spark, tgts)
    iterator = DummyIterator(train=train_sds)

    predict_col = "prediction"
    assembler = VectorAssembler(
        inputCols=train_sds.features,
        outputCol=f"_vassembler_features",
        handleInvalid="keep"
    )

    classifier = LightGBMClassifier(
        featuresCol=assembler.getOutputCol(),
        labelCol=train_sds.target_column,
        predictionCol=predict_col,
        metric="auc",
        validationIndicatorCol="tr_or_val",
        # isProvideTrainingMetric=True,
        # learningRate=0.05,
        # numLeaves=128,
        # featureFraction=0.9,
        # baggingFraction=0.9,
        # baggingFreq=1,
        # maxDepth=-1,
        # verbosity=-1,
        # minGainToSplit=0.0,
        # numThreads=0,
        # maxBin=255,
        # minDataInLeaf=3,
        # earlyStoppingRound=100,
        # numIterations=3000
    )

    train = train_sds.data\
        .join(train_sds.target, on=SparkDataset.ID_COLUMN)\
        .withColumn("tr_or_val", F.floor(F.rand(42) / 0.8).cast(BooleanType()))\
        .cache()
    test = test_sds.data.join(test_sds.target, on=SparkDataset.ID_COLUMN).cache()
    # train, test = train.randomSplit([0.8, 0.2], seed=42)
    train.count()
    test.count()

    ml_model = classifier.fit(assembler.transform(train))
    preds = ml_model.transform(assembler.transform(test))

    # spark_ml_algo = SparkBoostLGBM()
    # ml_model, _, _ = spark_ml_algo.fit_predict_single_fold(train_sds, train_sds)

    # preds = spark_ml_algo.predict_single_fold(test_sds, ml_model)
    # predict_col = 'prediction_LightGBM'

    # spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)
    # preds = spark_ml_algo.predict(test_sds)
    # preds = preds.data
    # predict_col = preds.features[0]

    pred_target_df = (
        preds
        # .join(train_sds.target, on=SparkDataset.ID_COLUMN, how='inner')
        .select(SparkDataset.ID_COLUMN, test_sds.target_column, predict_col)
    )

    pt_df = pred_target_df.toPandas()
    test_metric_value2 = evaluator(
        pt_df[train_sds.target_column].values,
        pt_df[predict_col].values
    )

    print(f"Test metric value2: {test_metric_value2}")
