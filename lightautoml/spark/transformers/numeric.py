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
from lightautoml.spark.transformers.base import SparkBaseEstimator, SparkBaseTransformer, ObsoleteSparkTransformer
from lightautoml.transformers.numeric import numeric_check


class SparkNaNFlagsEstimator(SparkBaseEstimator):
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    # TODO: the value is copied from the corresponding LAMA transformer.
    # TODO: it is better to be taken from shared module as a string constant
    _fname_prefix = "nanflg"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 do_replace_columns: bool = False,
                 nan_rate: float = 0.005):
        """

        Args:
            nan_rate: Nan rate cutoff.

        """
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=NumericRole(np.float32))
        self._nan_rate = nan_rate
        self._nan_cols: Optional[str] = None
        # self._features: Optional[List[str]] = None

    def _fit(self, sdf: SparkDataFrame) -> "Transformer":

        row = (
            sdf
            .select([F.mean(F.isnan(c).astype(FloatType())).alias(c) for c in self.getInputCols()])
            .first()
        )

        self._nan_cols = [col for col, col_nan_rate in row.asDict(True).items() if col_nan_rate > self._nan_rate]

        return SparkNaNFlagsTransformer(
            input_cols=self.getInputCols(),
            input_roles=self.getInputRoles(),
            output_cols=self._nan_cols,
            nan_cols=self._nan_cols
        )


class SparkNaNFlagsTransformer(SparkBaseTransformer):
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    # TODO: the value is copied from the corresponding LAMA transformer.
    # TODO: it is better to be taken from shared module as a string constant
    _fname_prefix = "nanflg"

    def __init__(self, 
                 input_cols: List[str],
                 input_roles: RolesDict,
                 output_cols: List[str],
                 nan_cols: List[str]):
        super().__init__(
            input_cols=input_cols,
            output_cols=output_cols,
            input_roles=input_roles,
            output_roles={f: NumericRole(np.float32) for f in output_cols},
            do_replace_columns=False)
        self._nan_cols = nan_cols

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:

        new_cols = [
            F.isnan(c).astype(FloatType()).alias(f"{self._fname_prefix}__{c}")
            for c in self._nan_cols
        ]

        out_sdf = self._make_output_df(sdf, new_cols)

        return out_sdf


class SparkFillInfTransformer(SparkBaseTransformer):
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillinf"

    def __init__(self, 
                 input_cols: List[str],
                 input_roles: RolesDict,
                 do_replace_columns=False):
        output_cols = [f"{self._fname_prefix}__{feat}" for feat in input_cols]
        super().__init__(
            input_cols=input_cols,
            output_cols=output_cols,
            input_roles=input_roles,
            output_roles={f: NumericRole(np.float32) for f in output_cols},
            do_replace_columns=do_replace_columns)

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:
        def is_inf(col: str):
            return F.col(col).isin([F.lit("+Infinity").cast("double"), F.lit("-Infinity").cast("double")])

        new_cols = [
            F.when(is_inf(i), np.nan).otherwise(F.col(i)).alias(f"{self._fname_prefix}__{i}")
            for i in self.getInputCols()
        ]

        out_df = self._make_output_df(df, cols_to_add=new_cols)

        return out_df


class SparkFillnaMedianEstimator(SparkBaseEstimator):
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
        self._meds: Optional[Dict[str, float]] = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        """Approximately estimates medians.

        Args:
            dataset: SparkDataFrame with numerical features.

        Returns:
            Spark MLlib Transformer

        """

        row = sdf\
            .select([F.percentile_approx(c, 0.5).alias(c) for c in self.getInputCols()])\
            .select([F.when(F.isnan(c), 0).otherwise(F.col(c)).alias(c) for c in self.getInputCols()])\
            .first()

        self._meds = row.asDict()

        return SparkFillnaMedianTransformer(input_cols=self.getInputCols(),
                                            output_cols=self.getOutputCols(),
                                            input_roles=self.getInputRoles(),
                                            output_roles=self.getOutputRoles(),
                                            meds=self._meds)


class SparkFillnaMedianTransformer(SparkBaseTransformer):
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

        new_cols = [
            F.when(F.isnan(c), self._meds[c]).otherwise(F.col(c)).alias(f"{self._fname_prefix}__{c}")
            for c in self.getInputCols()
        ]

        out_sdf = self._make_output_df(sdf, new_cols)

        return out_sdf


class SparkLogOddsTransformer(SparkBaseTransformer):
    """Convert probs to logodds."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "logodds"

    _can_unwind_parents = False

    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict,
                 do_replace_columns=False):
        super().__init__(input_cols=input_cols,
                         output_cols=[f"{self._fname_prefix}__{feat}" for feat in input_cols],
                         input_roles=input_roles,
                         output_roles={f: NumericRole(np.float32) for f in input_cols},
                         do_replace_columns=do_replace_columns)

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform - convert num values to logodds.

        Args:
            dataset: SparkDataFrame dataset of categorical features.

        Returns:
            SparkDataFrame with encoded labels.

        """
        new_cols = []
        for i in self.getInputCols():
            col = F.when(F.col(i) < 1e-7, 1e-7) \
                .when(F.col(i) > 1 - 1e-7, 1 - 1e-7) \
                .otherwise(F.col(i))
            col = F.log(col / (F.lit(1) - col))
            new_cols.append(col.alias(f"{self._fname_prefix}__{i}"))

        out_sdf = self._make_output_df(sdf, new_cols)

        return out_sdf


class SparkStandardScalerEstimator(SparkBaseEstimator):
    """Classic StandardScaler."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "scaler"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 do_replace_columns: bool = False):
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=NumericRole(np.float32))
        self._means_and_stds: Optional[Dict[str, float]] = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        """Estimate means and stds.

        Args:
            sdf: SparkDataFrame of categorical features.

        Returns:
            StandardScalerTransformer instance

        """

        means = [F.mean(c).alias(f"mean_{c}") for c in self.getInputCols()]
        stds = [
            F.when(F.stddev(c) == 0, 1).when(F.isnan(F.stddev(c)), 1).otherwise(F.stddev(c)).alias(f"std_{c}")
            for c in self.getInputCols()
        ]

        self._means_and_stds = sdf.select(means + stds).first().asDict()

        return SparkStandardScalerTransformer(input_cols=self.getInputCols(),
                                              output_cols=self.getOutputCols(),
                                              input_roles=self.getInputRoles(),
                                              output_roles=self.getOutputRoles(),
                                              means_and_stds=self._means_and_stds)


class SparkStandardScalerTransformer(SparkBaseTransformer):
    """Classic StandardScaler."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "scaler"

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 means_and_stds: Dict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols=input_cols,
                         output_cols=output_cols,
                         input_roles=input_roles,
                         output_roles=output_roles,
                         do_replace_columns=do_replace_columns)
        self._means_and_stds = means_and_stds

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Scale test data.

        Args:
            sdf: SparkDataFrame of numeric features.

        Returns:
            SparkDataFrame with encoded labels.

        """

        new_cols = []
        for c in self.getInputCols():
            col = (F.col(c) - self._means_and_stds[f"mean_{c}"]) / F.lit(self._means_and_stds[f"std_{c}"])
            new_cols.append(col.alias(f"{self._fname_prefix}__{c}"))

        out_sdf = self._make_output_df(sdf, new_cols)

        return out_sdf


class SparkQuantileBinningEstimator(SparkBaseEstimator):
    """Discretization of numeric features by quantiles."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "qntl"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 do_replace_columns: bool = False,
                 nbins: int = 10):
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=CategoryRole(np.int32, label_encoded=True))
        self._nbins = nbins
        self._bucketizer = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        qdisc = QuantileDiscretizer(numBucketsArray=[self.nbins for _ in self.getInputCols()],
                                    handleInvalid="keep",
                                    inputCols=self.getInputCols(),
                                    outputCols=self.getOutputCols())

        self._bucketizer = qdisc.fit(sdf)

        return SparkQuantileBinningTransformer(
            self._nbins,
            self._bucketizer,
            input_cols=self.getInputCols(),
            input_roles=self.getInputRoles(),
            output_cols=self.getOutputCols(),
            output_roles=self.getOutputRoles()
        )


class SparkQuantileBinningTransformer(SparkBaseTransformer):
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "qntl"

    def __init__(self,
                 bins,
                 bucketizer,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols=input_cols,
                         output_cols=output_cols,
                         input_roles=input_roles,
                         output_roles=output_roles,
                         do_replace_columns=do_replace_columns)
        self._bins = bins
        self._bucketizer = bucketizer

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        new_cols =[
            F.when(F.col(c).astype(IntegerType()) == F.lit(self._bins), 0).otherwise(F.col(c).astype(IntegerType()) + 1).alias(c)
            for c in self._bucketizer.getOutputCols()
        ]

        if self.getDoReplaceColumns():
            input_cols = set(self.getInputCols())
            cols_to_leave = [f for f in sdf.columns if f not in input_cols]
        else:
            cols_to_leave = self.getInputCols()

        out_sdf = (
            self._bucketizer
            .transform(sdf)
            .select(*cols_to_leave, *new_cols)
        )

        return out_sdf
