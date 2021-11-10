import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole

from lightautoml.transformers.categorical import LabelEncoder
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder

from . import compare_by_content, compare_by_metadata, spark


def test_label_encoder(spark: SparkSession):

    source_data = pd.read_csv("test_transformers/datasets/dataset_23_cmc.csv")

    ds = PandasDataset(source_data, roles={name: CategoryRole(np.int32) for name in source_data.columns})

    compare_by_content(spark, ds, LabelEncoder(), SparkLabelEncoder())

