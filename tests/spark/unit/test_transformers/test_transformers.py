import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole, TextRole
from lightautoml.spark.transformers.decomposition import PCATransformer as SparkPCATransformer
from lightautoml.spark.transformers.numeric import NaNFlags as SparkNaNFlags
from lightautoml.spark.transformers.text import TfidfTextTransformer as SparkTfidfTextTransformer
from lightautoml.transformers.decomposition import PCATransformer
from lightautoml.transformers.numeric import NaNFlags
from lightautoml.transformers.text import TfidfTextTransformer
from . import compare_by_content, compare_by_metadata

# Note:
# -s means no stdout capturing thus allowing one to see what happens in reality

# IMPORTANT !
# The test requires env variable PYSPARK_PYTHON to be set
# for example: PYSPARK_PYTHON=/home/<user>/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


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


def test_pca(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, 21.34, 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, 1.4],
        "c": [0.11, 12.67, 89.1, 56.1, -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    # we are doing here 'smoke test' to ensure that it even can run at all
    # and also a check for metadat validity: features, roles, shapes should be ok
    lama_ds, spark_ds = compare_by_metadata(
        spark, ds, PCATransformer(n_components=10), SparkPCATransformer(n_components=10)
    )

    spark_data: np.ndarray = spark_ds.data

    # doing minor content check
    assert all(spark_data.flatten()), f"Data should not contain None-s: {spark_data.flatten()}"


def test_tfidf_text_transformer(spark: SparkSession):
    param_defaults = {
        "min_df": 1.0,
        "max_df": 100.0,
        "max_features": 3
    }
    source_data = pd.DataFrame(data={
        "a": ["ipsen loren doloren" for _ in range(10)],
        "b": ["ipsen loren doloren" for _ in range(10)],
        "c": ["ipsen loren doloren" for _ in range(10)],
    })

    ds = PandasDataset(source_data, roles={name: TextRole() for name in source_data.columns})

    # we cannot compare by content because the formulas used by Spark and scikit is slightly different
    # see: https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1461
    # and: https://spark.apache.org/docs/latest/ml-features#tf-idf
    compare_by_metadata(spark, ds, TfidfTextTransformer(param_defaults), SparkTfidfTextTransformer(param_defaults))



