from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from pyspark.ml import Transformer
from pyspark.ml.feature import QuantileDiscretizer, Bucketizer
from pyspark.sql import functions as F
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import FloatType, IntegerType
from lightautoml.dataset.base import RolesDict

from lightautoml.dataset.roles import ColumnRole, NumericRole, CategoryRole
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.transformers.base import SparkBaseEstimator, SparkBaseTransformer, SparkTransformer
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
        self._features: Optional[List[str]] = None

    def _fit(self, dataset: SparkDataset) -> "NaNFlags":

        sdf = dataset.data

        row = sdf\
            .select([F.mean(F.isnan(c).astype(FloatType())).alias(c) for c in dataset.features])\
            .collect()[0]

        self.nan_cols = [col for col, col_nan_rate in row.asDict(True).items() if col_nan_rate > self.nan_rate]
        self._features = list(self.nan_cols)

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        sdf = dataset.data

        new_sdf = sdf.select(*dataset.service_columns, *[
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

    _can_unwind_parents = False

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data

        for i in dataset.features:
            df = df \
                .withColumn(i,
                            F.when(
                                F.col(i).isin([F.lit("+Infinity").cast("double"), F.lit("-Infinity").cast("double")]),
                                None)
                            .otherwise(F.col(i))
                            ) \
                .withColumnRenamed(i, f"{self._fname_prefix}__{i}")

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(np.float32))

        return output


class FillInfTransformer(SparkBaseTransformer):

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillinf"

    _can_unwind_parents = False

    def __init__(self, 
                 input_cols: List[str],
                 input_roles: RolesDict,
                 role: ColumnRole):
        super().__init__(
            input_cols=input_cols,
            output_cols=[f"{self._fname_prefix}__{feat}" for feat in input_cols],
            input_roles=input_roles,
            output_roles={f: NumericRole(np.float32) for f in input_cols},
            do_replace_columns=False)

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:

        cols_to_select = []
        for i in self.getInputCols():
            col = F.when(
                    F.col(i).isin([F.lit("+Infinity").cast("double"), F.lit("-Infinity").cast("double")]),
                    None) \
                .otherwise(F.col(i))
            cols_to_select.append(col.alias(f"{self._fname_prefix}__{i}"))

        df = df.select('*', *cols_to_select)

        return df

class FillnaMedian(SparkTransformer):
    """Fillna with median."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed"

    def __init__(self):
        self.meds: Optional[Dict[str, float]] = None

    def _fit(self, dataset: SparkDataset):
        """Approximately estimates medians.

        Args:
            dataset: SparkDataset with numerical features.

        Returns:
            self.

        """

        print("I'm in fit")
        sdf = dataset.data

        rows = sdf\
            .select([F.percentile_approx(c, 0.5).alias(c) for c in dataset.features])\
            .select([F.when(F.isnan(c), 0).otherwise(F.col(c)).alias(c) for c in dataset.features])\
            .collect()

        assert len(rows) == 1, f"Results count should be exactly 1, but it is {len(rows)}"

        self.meds = rows[0].asDict()

        self._features = [f"{self._fname_prefix}__{c}" for c in dataset.features]

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform - fillna with medians.

        Args:
            dataset: SparkDataset of numerical features

        Returns:
            SparkDataset with replaced NaN with medians

        """
        print("I'm in transform")
        sdf = dataset.data

        new_sdf = sdf.select(*dataset.service_columns, *[
            F.when(F.isnan(c), self.meds[c]).otherwise(F.col(c)).alias(f"{self._fname_prefix}__{c}")
            for c in dataset.features
        ])

        output = dataset.empty()
        output.set_data(new_sdf, self.features, NumericRole(np.float32))

        return output


class FillnaMedianEstimator(SparkBaseEstimator):
    """Fillna with median."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 do_replace_columns: bool = False):
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=NumericRole(np.float32))
        self.meds: Optional[Dict[str, float]] = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        """Approximately estimates medians.

        Args:
            dataset: SparkDataFrame with numerical features.

        Returns:
            Spark MLlib Transformer

        """

        rows = sdf\
            .select([F.percentile_approx(c, 0.5).alias(c) for c in self.getInputCols()])\
            .select([F.when(F.isnan(c), 0).otherwise(F.col(c)).alias(c) for c in self.getInputCols()])\
            .collect()

        assert len(rows) == 1, f"Results count should be exactly 1, but it is {len(rows)}"

        self.meds = rows[0].asDict()

        return FillnaMedianTransformer(input_cols=self.getInputCols(),
                                       output_cols=self.getOutputCols(),
                                       input_roles=self.getInputRoles(),
                                       output_roles=self.getOutputRoles(),
                                       meds=self.meds)


class FillnaMedianTransformer(SparkBaseTransformer):
    """Fillna with median."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed"

    def __init__(self, input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 meds: Dict,
                 do_replace_columns: bool = False,):
        super().__init__(input_cols=input_cols,
                         output_cols=output_cols,
                         input_roles=input_roles,
                         output_roles=output_roles,
                         do_replace_columns=do_replace_columns)
        self._meds = meds

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform - fillna with medians.

        Args:
            dataset: SparkDataFrame of numerical features

        Returns:
            SparkDataFrame with replaced NaN with medians

        """

        cols_to_select = []
        for c in self.getInputCols():
            col = F.when(F.isnan(c), self.meds[c]).otherwise(F.col(c))
            cols_to_select.append(col.alias(f"{self._fname_prefix}__{c}"))

        sdf = sdf.select('*', *cols_to_select)

        return sdf


class LogOdds(SparkTransformer):
    """Convert probs to logodds."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "logodds"

    _can_unwind_parents = False

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """

        sdf = dataset.data

        # # transform
        # # TODO: maybe np.exp and then cliping and logodds?
        # data = np.clip(data, 1e-7, 1 - 1e-7)
        # data = np.log(data / (1 - data))
        new_sdf = sdf.select(
            SparkDataset.ID_COLUMN,
            *[
                F.when(F.col(c) < 1e-7, 1e-7)
                .when(F.col(c) > 1 - 1e-7, 1 - 1e-7)
                .otherwise(F.col(c))
                .alias(c)
                for c in dataset.features
            ]
        )\
        .select(
            SparkDataset.ID_COLUMN,
            *[F.log(F.col(c) / (F.lit(1) - F.col(c))).alias(f"{self._fname_prefix}__{c}")
              for c in dataset.features]
        )

        # create resulted
        output = dataset.empty()
        output.set_data(new_sdf, self.features, NumericRole(np.float32))

        return output


class LogOddsTransformer(SparkBaseTransformer):
    """Convert probs to logodds."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "logodds"

    _can_unwind_parents = False

    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict):
        super().__init__(input_cols=input_cols,
                         output_cols=[f"{self._fname_prefix}__{feat}" for feat in input_cols],
                         input_roles=input_roles,
                         output_roles={f: NumericRole(np.float32) for f in input_cols},
                         do_replace_columns=False)

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform - convert num values to logodds.

        Args:
            dataset: SparkDataFrame dataset of categorical features.

        Returns:
            SparkDataFrame with encoded labels.

        """

        # # transform
        # # TODO: maybe np.exp and then cliping and logodds?
        # data = np.clip(data, 1e-7, 1 - 1e-7)
        # data = np.log(data / (1 - data))

        cols_to_select = []
        for i in self.getInputCols():
            col = F.when(F.col(i) < 1e-7, 1e-7) \
                .when(F.col(i) > 1 - 1e-7, 1 - 1e-7) \
                .otherwise(F.col(i))
            col = F.log(col / (F.lit(1) - col))
            cols_to_select.append(col.alias(f"{self._fname_prefix}__{i}"))

        sdf = sdf.select('*', *cols_to_select)

        return sdf


class StandardScaler(SparkTransformer):
    """Classic StandardScaler."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "scaler"

    def __init__(self):
        super().__init__()
        self._means_and_stds: Optional[Dict[str, float]] = None

    def _fit(self, dataset: SparkDataset):
        """Estimate means and stds.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        sdf = dataset.data

        means = [F.mean(c).alias(f"mean_{c}") for c in sdf.columns]
        stds = [
            F.when(F.stddev(c) == 0, 1).when(F.isnan(F.stddev(c)), 1).otherwise(F.stddev(c)).alias(f"std_{c}")
            for c in dataset.features
        ]

        self._means_and_stds = sdf\
            .select(means + stds)\
            .collect()[0].asDict()

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Scale test data.

        Args:
            dataset: Pandas or Numpy dataset of numeric features.

        Returns:
            Numpy dataset with encoded labels.

        """

        sdf = dataset.data

        new_sdf = sdf.select(*dataset.service_columns, *[
            ((F.col(c) - self._means_and_stds[f"mean_{c}"]) / F.lit(self._means_and_stds[f"std_{c}"])).alias(f"{self._fname_prefix}__{c}")
            for c in dataset.features
        ])

        # create resulted
        output = dataset.empty()
        output.set_data(new_sdf, self.features, NumericRole(np.float32))

        return output


class QuantileBinning(SparkTransformer):
    """Discretization of numeric features by quantiles."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "qntl"

    def __init__(self, nbins: int = 10):
        """

        Args:
            nbins: maximum number of bins.
                One more will be added to keep NaN values (bin num = 0)

        """
        self.nbins = nbins
        self._bucketizer: Optional[Bucketizer] = None

    def _fit(self, dataset: SparkDataset):
        """Estimate bins borders.

        Args:
            dataset: Pandas or Numpy dataset of numeric features.

        Returns:
            self.

        """

        sdf = dataset.data

        qdisc = QuantileDiscretizer(numBucketsArray=[self.nbins for _ in dataset.features],
                                    handleInvalid="keep",
                                    inputCols=[c for c in dataset.features],
                                    outputCols=[f"{self._fname_prefix}__{c}" for c in dataset.features])

        self._bucketizer = qdisc.fit(sdf)

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Apply bin borders.

        Args:
            dataset: Spark dataset of numeric features.

        Returns:
            Spark dataset with encoded labels.

        """

        sdf = dataset.data

        # we do the last select to renumerate our bins
        # 0 bin is reserved for NaN's and the rest just shifted by +1
        # ...or keep (keep invalid values in a special additional bucket)
        # see: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.QuantileDiscretizer.html
        new_sdf = self._bucketizer\
            .transform(sdf)\
            .select(*dataset.service_columns, *[F.col(c).astype(IntegerType()).alias(c) for c in self._bucketizer.getOutputCols()])\
            .select(*dataset.service_columns, *[
                F.when(F.col(c) == self.nbins, 0).otherwise(F.col(c) + 1).alias(c)
                for c in self._bucketizer.getOutputCols()
            ])

        output = dataset.empty()
        output.set_data(new_sdf, self.features, CategoryRole(np.int32, label_encoded=True))

        return output
