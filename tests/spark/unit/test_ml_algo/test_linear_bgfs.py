import pickle

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS
from lightautoml.tasks import Task
from lightautoml.validation.base import DummyIterator
from ..test_transformers import from_pandas_to_spark

import pandas as pd


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.master("local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def test_smoke_linear_bgfs(spark: SparkSession):
    with open("unit/test_ml_algo/datasets/Lvl_0_Pipe_0_apply_selector.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    nds = NumpyDataset(data, features, roles, task=Task("binary"))
    pds = nds.to_pandas()
    target = pd.Series(target)

    sds = from_pandas_to_spark(pds, spark, target)
    iterator = DummyIterator(train=sds)

    ml_algo = LinearLBFGS()
    pred_ds = ml_algo.fit_predict(iterator)

    predicted_sdf = pred_ds.data
    predicted_sdf.show(10)

    assert SparkDataset.ID_COLUMN in predicted_sdf.columns
    assert len(pred_ds.features) == 1
    assert pred_ds.features[0].endswith("_prediction")
    assert pred_ds.features[0] in predicted_sdf.columns

####################################
# import os
# os.environ['PYSPARK_PYTHON'] = '/home/nikolay/.conda/envs/LAMA/bin/python'
#
# from pyspark.sql import SparkSession
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.linalg import Vectors
# from pyspark.sql import functions as F
#
# spark = SparkSession.builder.master("local[1]").getOrCreate()
#
# df = spark.createDataFrame([
#     (1.0, 2.0, Vectors.dense(1.0)),
#     (0.0, 2.0, Vectors.sparse(1, [], []))], ["label", "weight", "features"])
# lr = LinearRegression(regParam=0.0, solver="normal", weightCol="weight")
# lr.setMaxIter(5)
# lr.setRegParam(0.1)
#
# model = lr.fit(df)
# model.setFeaturesCol("features")
# model.setPredictionCol("newPrediction")
#
# test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
# model.transform(test0)
# abs(model.predict(test0.head().features) - (-1.0)) < 0.001
# True
# abs(model.transform(test0).head().newPrediction - (-1.0)) < 0.001
# True
# abs(model.coefficients[0] - 1.0) < 0.001
# True
# abs(model.intercept - 0.0) < 0.001

