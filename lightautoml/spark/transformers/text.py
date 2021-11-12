from typing import Optional

import numpy as np
from pyspark.ml.param import Param
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_list, concat, concat_ws
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Tokenizer as PysparkTokenizer
from pyspark.ml.feature import RegexTokenizer as PysparkRegexTokenizer

from lightautoml.dataset.roles import TextRole
from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.text import text_check
from lightautoml.text.tokenizer import BaseTokenizer
from lightautoml.text.tokenizer import SimpleEnTokenizer


class Tokenizer(SparkTransformer):
    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenized"

    def __init__(self, tokenizer: BaseTokenizer = SimpleEnTokenizer()):
        """
        Args:
            tokenizer: text tokenizer.
        """
        self.tokenizer = tokenizer

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        spark_data_frame = dataset.data
        spark_column_names = spark_data_frame.schema.names

        for i, column in enumerate(spark_column_names):
            # PysparkTokenizer transforms strings to lowercase, do not use it
            # tokenizer = PysparkTokenizer(inputCol=column, outputCol=self._fname_prefix + "__" + column)

            tokenizer = PysparkRegexTokenizer(inputCol=column, outputCol=self._fname_prefix + "__" + column, toLowercase=False)
            tokenized = tokenizer.transform(spark_data_frame)
            spark_data_frame = tokenized
            spark_data_frame = spark_data_frame.drop(column)

        output = dataset.empty()
        output.set_data(spark_data_frame, self.features, TextRole(np.str))

        return output


class ConcatTextTransformer(SparkTransformer):
    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "concated"

    def __init__(self, special_token: str = " [SEP] "):
        """

        Args:
            special_token: Add special token between columns.

        """
        self.special_token = special_token

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        spark_data_frame = dataset.data
        spark_column_names = spark_data_frame.schema.names

        colum_name = self._fname_prefix + "__" + "__".join(spark_column_names)
        concatExpr = concat_ws(self.special_token, *spark_column_names).alias(colum_name)
        concated = spark_data_frame.select(concatExpr)

        output = dataset.empty()
        output.set_data(concated, self.features, TextRole(np.str))

        return output
