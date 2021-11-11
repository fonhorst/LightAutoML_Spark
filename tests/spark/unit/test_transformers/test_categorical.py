from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole

from lightautoml.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder, \
    FreqEncoder as SparkFreqEncoder, OrdinalEncoder as SparkOrdinalEncoder

from . import compare_by_content, compare_by_metadata, spark


class DatasetForTest:

    def __init__(self, path: str,
                 columns: Optional[List[str]] = None,
                 roles: Optional[Dict] = None,
                 default_role: Optional[CategoryRole] = None):

        self.dataset = pd.read_csv(path)
        if columns is not None:
            self.dataset = self.dataset[columns]

        if roles is None:
            self.roles = {name: default_role for name in self.dataset.columns}
        else:
            self.roles = roles


DATASETS = [

    DatasetForTest("test_transformers/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("test_transformers/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32)
                   })
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_label_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, LabelEncoder(), SparkLabelEncoder())


@pytest.mark.parametrize("dataset", DATASETS)
def test_freq_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, FreqEncoder(), SparkFreqEncoder())


@pytest.mark.parametrize("dataset", DATASETS)
def test_ordinal_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, OrdinalEncoder(), SparkOrdinalEncoder())
