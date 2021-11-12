from typing import Optional, List, Dict
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.transformers.datetime import TimeToNum, BaseDiff, DateSeasons
from lightautoml.spark.transformers.datetime import TimeToNum as SparkTimeToNum, BaseDiff as SparkBaseDiff, \
    DateSeasons as SparkDateSeasons

from . import compare_by_content, compare_by_metadata, spark


class DatasetForTest:

    def __init__(self, path: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 columns: Optional[List[str]] = None,
                 roles: Optional[Dict] = None,
                 default_role: Optional[DatetimeRole] = None):

        if path is not None:
            self.dataset = pd.read_csv(path)
        else:
            self.dataset = df

        if columns is not None:
            self.dataset = self.dataset[columns]

        if roles is None:
            self.roles = {name: default_role for name in self.dataset.columns}
        else:
            self.roles = roles


DATASETS = [

    DatasetForTest(df=pd.DataFrame(data={
        "night": [
            "2000-01-01 00:00:00",
            np.nan,
            "2020-01-01 00:00:00",
            "2025-01-01 00:00:00",
            "2100-01-01 00:00:00",
        ],
        "morning": [
            "2000-01-01 06:00:00",
            "2017-01-01 06:00:00",
            "2020-01-01 06:00:00",
            None,
            "2100-01-01 06:00:00",
        ],
        "day": [
            np.nan,
            "2017-01-01 12:00:00",
            "2020-01-01 12:00:00",
            "2025-01-01 12:00:00",
            "2100-01-01 12:00:00",
        ],
        "evening": [
            "2000-01-01 20:00:00",
            "2017-01-01 20:00:00",
            "2020-01-01 20:00:00",
            "2025-01-01 20:00:00",
            "2100-01-01 20:00:00",
        ],
    }), default_role=DatetimeRole()),

    DatasetForTest(df=pd.DataFrame(data={
        "night": [
            "2000-06-05 00:00:00",
            "2020-01-01 00:00:00",
            "2025-05-01 00:00:00",
            "2100-08-01 00:00:00",
        ],
        "morning": [
            "2000-03-01 06:00:00",
            "2017-02-01 06:00:00",
            "2020-04-01 06:00:00",
            "2100-11-01 06:00:00",
        ],
        "day": [
            "2017-05-01 12:00:00",
            "2020-02-01 12:00:00",
            "2025-01-01 12:00:00",
            "2100-01-01 12:00:00",
        ],
        "evening": [
            "2000-01-01 20:00:00",
            "2020-01-01 20:00:00",
            "2025-01-01 20:00:00",
            "2100-01-01 20:00:00",
        ],
    }), default_role=DatetimeRole(country="Russia")),

]


@pytest.mark.parametrize("dataset", DATASETS)
def test_time_to_num(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, TimeToNum(), SparkTimeToNum())


@pytest.mark.parametrize("dataset", DATASETS)
def test_base_diff(spark: SparkSession, dataset: DatasetForTest):

    columns: List[str] = dataset.dataset.columns
    middle = int(len(columns)/2)
    base_names = columns[:middle]
    diff_names = columns[middle:]

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(
        spark,
        ds,
        BaseDiff(base_names=base_names, diff_names=diff_names),
        SparkBaseDiff(base_names=base_names, diff_names=diff_names)
    )


@pytest.mark.parametrize("dataset", [DATASETS[1]])
def test_date_seasons(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, DateSeasons(), SparkDateSeasons())
