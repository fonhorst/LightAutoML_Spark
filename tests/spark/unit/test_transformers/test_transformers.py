import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole, TextRole
from lightautoml.spark.transformers.decomposition import PCATransformer as SparkPCATransformer
from lightautoml.spark.transformers.numeric import NaNFlags as SparkNaNFlags
from lightautoml.transformers.decomposition import PCATransformer
from lightautoml.transformers.numeric import NaNFlags
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


def test_tokenizer(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": ["This function is intended to compare", "two DataFrames and", "output any differences"],
        "b": ["Is is mostly intended ", "for use in unit tests", "Additional parameters allow "],
        "c": ["varying the strictness", "of the equality ", "checks performed"],
        "d": ["This example shows comparing", "two DataFrames that are equal", "but with columns of differing dtypes"]
    })

    ds = PandasDataset(source_data, roles={name: TextRole(np.str) for name in source_data.columns})

    from lightautoml.transformers.text import TokenizerTransformer
    from lightautoml.transformers.text import SimpleEnTokenizer

    lama_tokenizer_transformer = TokenizerTransformer(SimpleEnTokenizer(is_stemmer=False,to_string=False))
    lama_tokenizer_transformer.fit(ds)
    lama_result = lama_tokenizer_transformer.transform(ds)
    lama_result = lama_result.data
    print()
    print("lama_result")
    print(lama_result)

    from lightautoml.spark.transformers.text import Tokenizer as SparkTokenizer
    from lightautoml.spark.utils import from_pandas_to_spark

    spark_tokenizer_transformer = SparkTokenizer()
    spark_dataset = from_pandas_to_spark(ds, spark)
    spark_tokenizer_transformer.fit(spark_dataset)
    spark_result = spark_tokenizer_transformer.transform(spark_dataset)
    spark_result = spark_result.to_pandas().data
    print("spark_result")
    print(spark_result)

    from pandas._testing import assert_frame_equal
    assert_frame_equal(lama_result, spark_result)


    # compare_by_content(spark, ds, lama_tokenizer_transformer, spark_tokenizer_transformer)


def test_concat_text_transformer(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": ["This function is intended to compare", "two DataFrames and", "output any differences"],
        "b": ["Is is mostly intended ", "for use in unit tests", "Additional parameters allow "],
        "c": ["varying the strictness", "of the equality ", "checks performed"],
        "d": ["This example shows comparing", "two DataFrames that are equal", "but with columns of differing dtypes"]
    })

    ds = PandasDataset(source_data, roles={name: TextRole(np.str) for name in source_data.columns})

    from lightautoml.transformers.text import ConcatTextTransformer

    lama_transformer = ConcatTextTransformer()
    lama_transformer.fit(ds)
    lama_result = lama_transformer.transform(ds)
    lama_result = lama_result.data
    print()
    print("lama_result:")
    print(lama_result)

    from lightautoml.spark.transformers.text import ConcatTextTransformer as SparkConcatTextTransformer
    from lightautoml.spark.utils import from_pandas_to_spark

    spark_tokenizer_transformer = SparkConcatTextTransformer()
    spark_dataset = from_pandas_to_spark(ds, spark)
    spark_tokenizer_transformer.fit(spark_dataset)
    spark_result = spark_tokenizer_transformer.transform(spark_dataset)
    spark_result = spark_result.to_pandas().data
    print()
    print("spark_result:")
    print(spark_result)

    from pandas._testing import assert_frame_equal
    assert_frame_equal(lama_result, spark_result)


    # compare_by_content(spark, ds, lama_tokenizer_transformer, spark_tokenizer_transformer)


