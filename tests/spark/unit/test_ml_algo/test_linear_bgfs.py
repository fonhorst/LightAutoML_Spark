import pickle

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS
from lightautoml.tasks import Task
from lightautoml.validation.base import DummyIterator
from ..test_transformers import from_pandas_to_spark


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def test_smoke_linear_bgfs(spark: SparkSession):
    with open("test_ml_algo/datasets/Lvl_0_Pipe_0_apply_selector.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    nds = NumpyDataset(data, features, roles, task=Task("binary"))
    pds = nds.to_pandas()

    iterator = DummyIterator(train=from_pandas_to_spark(pds, spark, target))

    ml_algo = LinearLBFGS()
    predicted = ml_algo.fit_predict(iterator).data

    predicted.show(10)