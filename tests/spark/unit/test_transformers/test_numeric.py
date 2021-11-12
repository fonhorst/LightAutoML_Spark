from typing import Optional, List, Dict
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.transformers.numeric import NaNFlags as SparkNaNFlags, FillInf as SparkFillInf, \
    FillnaMedian as SparkFillnaMedian
from lightautoml.transformers.numeric import NaNFlags, FillInf, FillnaMedian

from . import compare_by_content, compare_by_metadata, spark

# Note:
# -s means no stdout capturing thus allowing one to see what happens in reality

# IMPORTANT !
# The test requires env variable PYSPARK_PYTHON to be set
# for example: PYSPARK_PYTHON=/home/<user>/.conda/envs/LAMA/bin/python


class DatasetForTest:

    def __init__(self, path: str,
                 columns: Optional[List[str]] = None,
                 roles: Optional[Dict] = None,
                 default_role: Optional[NumericRole] = None):

        self.dataset = pd.read_csv(path)
        if columns is not None:
            self.dataset = self.dataset[columns]

        if roles is None:
            self.roles = {name: default_role for name in self.dataset.columns}
        else:
            self.roles = roles


DATASETS = [

    DatasetForTest("test_transformers/datasets/dataset_23_cmc.csv", default_role=NumericRole(np.int32)),

    DatasetForTest("test_transformers/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "LotFrontage"],
                   roles={
                       "Id": NumericRole(np.int32),
                       "MSSubClass": NumericRole(np.int32),
                       "LotFrontage": NumericRole(np.float32)
                   })
]


def test_nan_flags(spark: SparkSession):
    nan_rate = 0.2
    source_data = pd.DataFrame(data={
        "a": [None if i >= 5 else i for i in range(10)],
        "b": [None if i >= 7 else i for i in range(10)],
        "c": [None if i == 2 else i for i in range(10)],
        "d": list(range(10))
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    compare_by_content(spark, ds, NaNFlags(nan_rate), SparkNaNFlags(nan_rate))


@pytest.mark.parametrize("dataset", DATASETS)
def test_fill_inf(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, FillInf(), SparkFillInf())


@pytest.mark.parametrize("dataset", DATASETS)
def test_fillna_median(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, FillnaMedian(), SparkFillnaMedian())
