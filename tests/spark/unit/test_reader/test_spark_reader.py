import pytest
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.tasks import Task
from . import spark
import pandas as pd


@pytest.mark.parametrize("cv", [1, 5, 10])
def test_spark_reader(spark: SparkSession, cv: int):
    df = spark.read.csv("../../examples/data/sampled_app_train.csv", header=True)
    sreader = SparkToSparkReader(task=Task("binary"), cv=cv)
    sds = sreader.fit_read(df)

    # 1. it should have _id
    # 2. it should have target
    # 3. it should have roles for all columns

    assert sds.target_column not in sds.data.columns
    assert isinstance(sds.target, SparkDataFrame) \
           and sds.target_column in sds.target.columns \
           and SparkDataset.ID_COLUMN in sds.target.columns
    assert SparkDataset.ID_COLUMN in sds.data.columns
    assert set(sds.features).issubset(sds.roles.keys())
    assert all(f in sds.data.columns for f in sds.features)
    assert "folds" in sds.__dict__ and sds.folds

    assert len(sds.folds) == cv

    correct_folds = (
        (SparkDataset.ID_COLUMN in train.columns) and (SparkDataset.ID_COLUMN in val.columns)
        for train, val in sds.folds
    )

    assert all(correct_folds), "ID_COLUMN should be presented everywhere"


