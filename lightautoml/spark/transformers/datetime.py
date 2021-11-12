from typing import Optional, Sequence
from collections import defaultdict
from itertools import chain, combinations
import numpy as np
from pandas import Series
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame

from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.transformers.datetime import datetime_check, date_attrs

from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer


class TimeToNum(SparkTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference
    with basic_date (``basic_date == '2020-01-01'``).
    """

    basic_time = "2020-01-01"

    basic_interval = "D"

    _interval_mapping = {
        "NS": 0.000000001,
        "MS": 0.001,
        "SEC": 1,
        "MIN": 60,
        "HOUR": 60*60,
        "D": 60*60*24,

        # FIXME SPARK-LAMA: Very rough rounding
        "M": 60*60*24*30,
        "Y": 60*60*24*365
    }

    _fname_prefix = "dtdiff"
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        df = dataset.data

        for i in df.columns:
            df = df.withColumn(
                i,
                (
                    F.to_timestamp(F.col(i)).cast("long") - F.to_timestamp(F.lit(self.basic_time)).cast("long")
                ) / self._interval_mapping[self.basic_interval]
            ).withColumnRenamed(i, f"{self._fname_prefix}__{i}")

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(np.float32))

        return output

