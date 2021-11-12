from typing import Optional

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Imputer

from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.numeric import numeric_check


class NaNFlags(SparkTransformer):
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    # TODO: the value is copied from the corresponding LAMA transformer.
    # TODO: it is better to be taken from shared module as a string constant
    _fname_prefix = "nanflg"

    def __init__(self, nan_rate: float = 0.005):
        """

        Args:
            nan_rate: Nan rate cutoff.

        """
        self.nan_rate = nan_rate
        self.nan_cols: Optional[str] = None
        self._features: Optional[str] = None

    def fit(self, dataset: SparkDataset) -> "NaNFlags":
        # TODO: can be in the base class (SparkTransformer)
        for check_func in self._fit_checks:
            check_func(dataset)

        sdf = dataset.data

        row = sdf\
            .select([F.mean(F.isnan(c).astype(FloatType())).alias(c) for c in sdf.columns])\
            .collect()[0]

        self.nan_cols = [col for col, col_nan_rate in row.asDict(True).items() if col_nan_rate > self.nan_rate]
        self._features = list(self.nan_cols)

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        # TODO: can be in the base class (SparkTransformer)
        # checks here
        super().transform(dataset)

        sdf = dataset.data

        new_sdf = sdf.select([
            F.isnan(c).astype(FloatType()).alias(feat)
            for feat, c in zip(self.features, self.nan_cols)
        ])

        output = dataset.empty()
        output.set_data(new_sdf, self.features, NumericRole(np.float32))

        return output


class FillInf(SparkTransformer):

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillinf"

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        df = dataset.data

        for i in df.columns:
            df = df \
                .withColumn(i,
                            F.when(
                                F.col(i).isin([F.lit("+Infinity").cast("double"), F.lit("-Infinity").cast("double")]),
                                np.nan)
                            .otherwise(F.col(i))
                            ) \
                .withColumnRenamed(i, f"{self._fname_prefix}__{i}")

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(np.float32))

        return output


class FillnaMedian(SparkTransformer):

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed"

    def fit(self, dataset: SparkDataset) -> "FillnaMedian":

        super().fit(dataset)
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        df = dataset.data

        imputer = Imputer(
            inputCols=df.columns,
            outputCols=df.columns
        ).setStrategy("median")

        df = imputer.fit(df).transform(df)

        for i in df.columns:
            df = df.withColumnRenamed(i, f"{self._fname_prefix}__{i}")

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(np.float32))

        return output

