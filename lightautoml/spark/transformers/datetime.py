from typing import Optional, Sequence, List
from collections import defaultdict
from itertools import chain, combinations
import numpy as np
from pandas import Series
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame

from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.transformers.datetime import datetime_check, date_attrs

from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer


class SparkDatetimeTransformer(SparkTransformer):

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

    _fit_checks = (datetime_check,)
    _transform_checks = ()


class TimeToNum(SparkDatetimeTransformer):

    basic_time = "2020-01-01"
    _fname_prefix = "dtdiff"

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


class BaseDiff(SparkDatetimeTransformer):

    _fname_prefix = "basediff"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self,
                 base_names: Sequence[str],
                 diff_names: Sequence[str],
                 basic_interval: Optional[str] = "D"):

        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

    def fit(self, dataset: SparkDataset) -> "SparkTransformer":

        # FIXME SPARK-LAMA: Возможно это можно будет убрать, т.к. у датасета будут колонки
        self._features = []
        for col in self.base_names:
            self._features.extend([f"{self._fname_prefix}_{col}__{x}" for x in self.diff_names])

        for check_func in self._fit_checks:
            check_func(dataset)

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        df = dataset.data

        for dif in self.diff_names:
            for base in self.base_names:
                df = df.withColumn(
                    f"{self._fname_prefix}_{base}__{dif}",
                    (
                        F.to_timestamp(F.col(dif)).cast("long") - F.to_timestamp(F.col(base)).cast("long")
                    ) / self._interval_mapping[self.basic_interval]
                )

        df = df.select(
            [f"{self._fname_prefix}_{base}__{dif}" for base in self.base_names for dif in self.diff_names]
        )

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(dtype=np.float32))

        return output

