import logging
import pickle
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
from typing import Optional, Sequence, List, Tuple, Dict, Union, cast, Iterator, Any

import numpy as np
from pandas import Series
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame, Window, Column, SparkSession
from pyspark.sql.functions import udf, array, monotonically_increasing_id
from pyspark.sql.types import FloatType, DoubleType, IntegerType
from sklearn.utils.murmurhash import murmurhash3_32

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.mlwriters import TmpСommonMLWriter
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer, SparkBaseEstimator, SparkBaseTransformer
from lightautoml.spark.utils import get_cached_df_through_rdd, cache
from lightautoml.transformers.categorical import categorical_check, encoding_check, oof_task_check, \
    multiclass_task_check
from lightautoml.transformers.base import LAMLTransformer

from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.param.shared import TypeConverters, Param, Params
from pyspark.ml.util import MLReadable, MLWritable, MLWriter


logger = logging.getLogger(__name__)


# FIXME SPARK-LAMA: np.nan in str representation is 'nan' while Spark's NaN is 'NaN'. It leads to different hashes.
# FIXME SPARK-LAMA: If udf is defined inside the class, it not works properly.
# "if murmurhash3_32 can be applied to a whole pandas Series, it would be better to make it via pandas_udf"
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749534669
murmurhash3_32_udf = F.udf(lambda value: murmurhash3_32(value.replace("NaN", "nan"), seed=42) if value is not None else None, SparkTypes.IntegerType())


class TypesHelper:
    _ad_hoc_types_mapper = defaultdict(
        lambda: "string",
        {
            "bool": "boolean",
            "int": "int",
            "int8": "int",
            "int16": "int",
            "int32": "int",
            "int64": "int",
            "int128": "bigint",
            "int256": "bigint",
            "integer": "int",
            "uint8": "int",
            "uint16": "int",
            "uint32": "int",
            "uint64": "int",
            "uint128": "bigint",
            "uint256": "bigint",
            "longlong": "long",
            "ulonglong": "long",
            "float16": "float",
            "float": "float",
            "float32": "float",
            "float64": "double",
            "float128": "double"
        }
    )

    _spark_numeric_types = (
        SparkTypes.ShortType,
        SparkTypes.IntegerType,
        SparkTypes.LongType,
        SparkTypes.FloatType,
        SparkTypes.DoubleType,
        SparkTypes.DecimalType
    )


def pandas_dict_udf(broadcasted_dict):

    def f(s: Series) -> Series:
        values_dict = broadcasted_dict.value
        return s.map(values_dict)
    return F.pandas_udf(f, "double")


class SparkLabelEncoderEstimator(SparkBaseEstimator, TypesHelper):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 subs: Optional[float] = None,
                 random_state: Optional[int] = 42,
                 do_replace_columns: bool = False,
                 output_role: Optional[ColumnRole] = None):
        if not output_role:
            output_role = CategoryRole(np.int32, label_encoded=True)
        super().__init__(input_cols, input_roles, do_replace_columns=do_replace_columns, output_role=output_role)
        self._input_columns = self.getInputCols()
        self._input_roles = self.getInputRoles()

    def _fit(self, dataset: SparkDataFrame) -> "SparkLabelEncoderTransformer":
        logger.info(f"[{type(self)} (LE)] fit is started")

        roles = self._input_roles#self.getOrDefault(self.inputRoles)

        df = dataset

        self.dicts = dict()

        for i in self._input_columns:

            logger.debug(f"[{type(self)} (LE)] fit column {i}")

            role = roles[i]

            # TODO: think what to do with this warning
            co = role.unknown

            # FIXME SPARK-LAMA: Possible OOM point
            # TODO SPARK-LAMA: Can be implemented without multiple groupby and thus shuffling using custom UDAF.
            # May be an alternative it there would be performance problems.
            # https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749539901
            vals = df \
                .groupBy(i).count() \
                .where(F.col("count") > co) \
                .select(i, F.col("count")) \
                .toPandas()

            logger.debug(f"[{type(self)} (LE)] toPandas is completed")

            vals = vals.sort_values(["count", i], ascending=[False, True])
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])
            logger.debug(f"[{type(self)} (LE)] pandas processing is completed")

        logger.info(f"[{type(self)} (LE)] fit is finished")

        return SparkLabelEncoderTransformer(input_cols=self.getInputCols(),
                                            output_cols=self.getOutputCols(),
                                            input_roles=self.getInputRoles(),
                                            output_roles=self.getOutputRoles(),
                                            do_replace_columns=self.getDoReplaceColumns(),
                                            dicts=self.dicts)


class SparkLabelEncoderTransformer(SparkBaseTransformer, TypesHelper):
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    fittedDicts = Param(Params._dummy(), "fittedDicts", "dicts from fitted Estimator")

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self.set(self.fittedDicts, dicts)
        self._dicts = dicts
        self._input_columns = self.getInputCols()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        logger.info(f"[{type(self)} (LE)] transform is started")

        df = dataset
        sc = df.sql_ctx.sparkSession.sparkContext

        cols_to_select = []

        for i, out_col in zip(self._input_columns,  self.getOutputCols()):
            logger.debug(f"[{type(self)} (LE)] transform col {i}")

            _ic = F.col(i)

            if i not in self._dicts:
                col = _ic
            elif len(self._dicts[i]) == 0:
                col = F.lit(self._fillna_val)
            else:
                vals: dict = self._dicts[i].to_dict()

                null_value = self._fillna_val
                if None in vals:
                    null_value = vals[None]
                    _ = vals.pop(None, None)

                if len(vals) == 0:
                    col = F.when(_ic.isNull(), null_value).otherwise(None)
                else:

                    nan_value = self._fillna_val

                    # if np.isnan(list(vals.keys())).any():  # not working
                    # Вот этот кусок кода тут по сути из-за OrdinalEncoder, который
                    # в КАЖДЫЙ dicts пихает nan. И вот из-за этого приходится его отсюда чистить.
                    # Нужно подумать, как это всё зарефакторить.
                    new_dict = {}
                    for key, value in vals.items():
                        try:
                            if np.isnan(key):
                                nan_value = value
                            else:
                                new_dict[key] = value
                        except TypeError:
                            new_dict[key] = value

                    vals = new_dict

                    if len(vals) == 0:
                        col = F.when(F.isnan(_ic), nan_value).otherwise(None)
                    else:
                        logger.debug(f"[{type(self)} (LE)] map size: {len(vals)}")

                        labels = sc.broadcast(vals)

                        if type(df.schema[i].dataType) in self._spark_numeric_types:
                            col = F.when(_ic.isNull(), null_value) \
                                .otherwise(
                                    F.when(F.isnan(_ic), nan_value)
                                     .otherwise(pandas_dict_udf(labels)(_ic))
                                )
                        else:
                            col = F.when(_ic.isNull(), null_value) \
                                   .otherwise(pandas_dict_udf(labels)(_ic))

            cols_to_select.append(col.alias(out_col))

        logger.info(f"[{type(self)} (LE)] Transform is finished")

        output = self._make_output_df(df, cols_to_select).fillna(self._fillna_val)

        return output


class SparkOrdinalEncoderEstimator(SparkLabelEncoderEstimator):
    _fit_checks = (categorical_check,)
    _fname_prefix = "ord"
    _fillna_val = np.nan

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 subs: Optional[float] = None,
                 random_state: Optional[int] = 42):
        super().__init__(input_cols, input_roles, subs, random_state, output_role=NumericRole(np.float32))
        self.dicts = None
        self._use_cols = self.getInputCols()

    def _fit(self, dataset: SparkDataFrame) -> "Transformer":

        logger.info(f"[{type(self)} (ORD)] fit is started")

        # roles = dataset.roles
        roles = self.getOrDefault(self.inputRoles)

        self.dicts = {}
        for i in self._use_cols:

            logger.debug(f"[{type(self)} (ORD)] fit column {i}")

            role = roles[i]

            if not type(dataset.schema[i].dataType) in self._spark_numeric_types:

                co = role.unknown

                cnts = dataset \
                    .groupBy(i).count() \
                    .where((F.col("count") > co) & F.col(i).isNotNull()) \

                # TODO SPARK-LAMA: isnan raises an exception if column is boolean.
                if type(dataset.schema[i].dataType) != SparkTypes.BooleanType:
                    cnts = cnts \
                        .where(~F.isnan(F.col(i)))

                cnts = cnts \
                    .select(i) \
                    .toPandas()

                logger.debug(f"[{type(self)} (ORD)] toPandas is completed")

                cnts = Series(cnts[i].astype(str).rank().values, index=cnts[i])
                self.dicts[i] = cnts.append(Series([cnts.shape[0] + 1], index=[np.nan])).drop_duplicates()
                logger.debug(f"[{type(self)} (ORD)] pandas processing is completed")

        logger.info(f"[{type(self)} (ORD)] fit is finished")

        return SparkOrdinalEncoderTransformer(input_cols=self.getInputCols(),
                                              output_cols=self.getOutputCols(),
                                              input_roles=self.getInputRoles(),
                                              output_roles=self.getOutputRoles(),
                                              do_replace_columns=self.getDoReplaceColumns(),
                                              dicts=self.dicts)


class SparkOrdinalEncoderTransformer(SparkLabelEncoderTransformer):
    _transform_checks = ()
    _fname_prefix = "ord"
    _fillna_val = np.nan

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)


class SparkFreqEncoderEstimator(SparkLabelEncoderEstimator):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    _fillna_val = 1

    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, input_roles, do_replace_columns, output_role=NumericRole(np.float32))

    def _fit(self, dataset: SparkDataFrame) -> "SparkFreqEncoderTransformer":

        logger.info(f"[{type(self)} (FE)] fit is started")

        df = dataset

        self.dicts = {}
        for i in self.getInputCols():

            logger.info(f"[{type(self)} (FE)] fit column {i}")

            vals = df \
                .groupBy(i).count() \
                .where(F.col("count") > 1) \
                .select(i, F.col("count")) \
                .toPandas()

            logger.debug(f"[{type(self)} (FE)] toPandas is completed")

            self.dicts[i] = vals.set_index(i)["count"]

            logger.debug(f"[{type(self)} (LE)] pandas processing is completed")

        logger.info(f"[{type(self)} (FE)] fit is finished")

        return SparkFreqEncoderTransformer(input_cols=self.getInputCols(),
                                           output_cols=self.getOutputCols(),
                                           input_roles=self.getInputRoles(),
                                           output_roles=self.getOutputRoles(),
                                           do_replace_columns=self.getDoReplaceColumns(),
                                           dicts=self.dicts)


class SparkFreqEncoderTransformer(SparkLabelEncoderTransformer):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"
    _fillna_val = 1

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)


class SparkCatIntersectionsHelper:
    _fname_prefix = "inter"

    def _make_col_name(self, cols: Sequence[str]) -> str:
        return f"({'__'.join(cols)})"

    def _make_category(self, cols: Sequence[str]) -> Column:
        lit = F.lit("_")
        col_name = self._make_col_name(cols)
        columns_for_concat = []
        for col in cols:
            columns_for_concat.append(F.col(col))
            columns_for_concat.append(lit)
        columns_for_concat = columns_for_concat[:-1]

        return murmurhash3_32_udf(F.concat(*columns_for_concat)).alias(col_name)

    def _build_df(self, df: SparkDataFrame,
                  intersections: Optional[Sequence[Sequence[str]]]) -> SparkDataFrame:
        columns_to_select = [
            self._make_category(comb)
                .alias(f"{self._make_col_name(comb)}") for comb in intersections]
        df = df.select('*', *columns_to_select)
        return df


class SparkCatIntersectionsEstimator(SparkCatIntersectionsHelper, SparkLabelEncoderEstimator):

    _fit_checks = (categorical_check,)
    _transform_checks = ()

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 intersections: Optional[Sequence[Sequence[str]]] = None,
                 max_depth: int = 2,
                 do_replace_columns: bool = False):

        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=CategoryRole(np.int32, label_encoded=True))
        self.intersections = intersections
        self.max_depth = max_depth

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(self.getInputCols())) + 1):
                self.intersections.extend(list(combinations(self.getInputCols(), i)))

        self._input_roles = {
            f"{self._make_col_name(comb)}": CategoryRole(
                np.int32,
                unknown=max((self.getInputRoles()[x].unknown for x in comb)),
                label_encoded=True,
            ) for comb in self.intersections
        }
        self._input_columns = list(self._input_roles.keys())

        out_roles = {f"{self._fname_prefix}__{f}": role
                     for f, role in self._input_roles.items()}

        self.set(self.outputCols, list(out_roles.keys()))
        self.set(self.outputRoles, out_roles)

    def _fit(self, df: SparkDataFrame) -> Transformer:
        logger.info(f"[{type(self)} (CI)] fit is started")
        inter_df = self._build_df(df, self.intersections)

        super()._fit(inter_df)

        logger.info(f"[{type(self)} (CI)] fit is finished")

        return SparkCatIntersectionsTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.getInputRoles(),
            output_roles=self.getOutputRoles(),
            do_replace_columns=self.getDoReplaceColumns(),
            dicts=self.dicts,
            intersections=self.intersections
        )


class SparkCatIntersectionsTransformer(SparkCatIntersectionsHelper, SparkLabelEncoderTransformer):

    _fit_checks = (categorical_check,)
    _transform_checks = ()

    _fillna_val = 0

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict,
                 intersections: Optional[Sequence[Sequence[str]]] = None):

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)
        self.intersections = intersections

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:
        inter_df = self._build_df(df, self.intersections)
        temp_cols = set(inter_df.columns).difference(df.columns)
        self._input_columns = temp_cols

        out_df = super()._transform(inter_df)

        out_df = out_df.drop(*temp_cols)
        return out_df


class SparkOHEEncoderEstimator(SparkBaseEstimator):
    """
    Simple OneHotEncoder over label encoded categories.
    """

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        input_cols: List[str],
        input_roles: Dict[str, ColumnRole],
        do_replace_columns: bool = False,
        make_sparse: Optional[bool] = None,
        total_feats_cnt: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """

        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.

        """
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=None) # TODO: temporary stub, output roles are not calculated at this moment

        self._make_sparse = make_sparse
        self._total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self._make_sparse is None:
            assert self._total_feats_cnt is not None, "Param total_feats_cnt should be defined if make_sparse is None"

        self._ohe_transformer_and_roles: Optional[Tuple[Transformer, Dict[str, ColumnRole]]] = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        """Calc output shapes.

        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        maxs = [F.max(c).alias(f"max_{c}") for c in self.getInputCols()]
        mins = [F.min(c).alias(f"min_{c}") for c in self.getInputCols()]
        mm = sdf.select(maxs + mins).first().asDict()

        ohe = OneHotEncoder(inputCols=self.getInputCols(), outputCols=self.getOutputCols(), handleInvalid="error")
        transformer = ohe.fit(sdf)

        roles = {
            f"{self._fname_prefix}__{c}": NumericVectorOrArrayRole(
                size=mm[f"max_{c}"] - mm[f"min_{c}"] + 1,
                element_col_name_template=[
                    f"{self._fname_prefix}_{i}__{c}"
                    for i in np.arange(mm[f"min_{c}"], mm[f"max_{c}"] + 1)
                ]
            ) for c in self.getInputCols()
        }

        self._ohe_transformer_and_roles = (transformer, roles)

        return OHEEncoderTransformer(
            transformer,
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.getInputRoles(),
            output_roles=roles
        )


class OHEEncoderTransformer(SparkBaseTransformer):
    """OHEEncoder Transformer"""

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self,
                 ohe_transformer: Transformer,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._ohe_transformer = ohe_transformer

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        output = self._ohe_transformer.transform(sdf)

        return output


def te_mapping_udf(broadcasted_dict):
    def f(folds, current_column):
        values_dict = broadcasted_dict.value
        try:
            return values_dict[f"{folds}_{current_column}"]
        except KeyError:
            return np.nan
    return F.udf(f, "double")


class SparkTargetEncoderEstimator(SparkBaseEstimator):
    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0),
                 task_name: Optional[str] = None,
                 folds_column: Optional[str] = None,
                 target_column: Optional[str] = None,
                 do_replace_columns: bool = False
                ):
        super().__init__(input_cols, input_roles, do_replace_columns,
                         NumericRole(np.float32, prob=task_name == "binary"))
        self.alphas = alphas
        self._task_name = task_name
        self._folds_column = folds_column
        self._target_column = target_column

    @staticmethod
    def score_func_binary(target, candidate) -> float:
        return -(
            target * np.log(candidate) + (1 - target) * np.log(1 - candidate)
        )

    @staticmethod
    def score_func_reg(target, candidate) -> float:
        return (target - candidate) ** 2

    def _fit(self, dataset: SparkDataFrame) -> "SparkTargetEncoderTransformer":
        logger.info(f"[{type(self)} (TE)] fit_transform is started")

        assert self._target_column in dataset.columns, "Target should be presented in the dataframe"
        assert self._folds_column in dataset.columns, "Folds should be presented in the dataframe"

        self.encodings = []

        df = dataset
        sc = df.sql_ctx.sparkSession.sparkContext

        _fc = F.col(self._folds_column)
        _tc = F.col(self._target_column)

        # float, int, float
        prior, total_count, total_target_sum = df.agg(
            F.mean(_tc.cast("double")),
            F.count(_tc),
            F.sum(_tc).cast("double")
        ).first()

        logger.debug(f"[{type(self)} (TE)] statistics is calculated")

        # we assume that there is not many folds in our data
        folds_prior_pdf = df.groupBy(_fc).agg(
            ((total_target_sum - F.sum(_tc)) / (total_count - F.count(_tc))).alias("_folds_prior")
        ).collect()

        logger.debug(f"[{type(self)} (TE)] folds_prior is calculated")

        folds_prior_map = {fold: prior for fold, prior in folds_prior_pdf}

        # cols_to_select = []

        for col_name in self.getInputCols():

            logger.debug(f"[{type(self)} (TE)] column {col_name}")

            _cur_col = F.col(col_name)

            _agg = (
                df
                    .groupBy(_fc, _tc, _cur_col)
                    .agg(F.sum(_tc).alias("_psum"), F.count(_tc).alias("_pcount"))
                    .toPandas()
            )

            logger.debug(f"[{type(self)} (TE)] _agg is calculated")

            candidates_pdf = _agg.groupby(
                by=[self._folds_column, col_name]
            )["_psum", "_pcount"].sum().reset_index().rename(columns={"_psum": "_fsum", "_pcount": "_fcount"})

            t_df = candidates_pdf.groupby(col_name).agg(
                _tsum=('_fsum', 'sum'),
                _tcount=('_fcount', 'sum')
            ).reset_index()

            candidates_pdf_2 = candidates_pdf.merge(t_df, on=col_name, how='inner')

            def make_candidates(x):
                cat_val, fold, fsum, tsum, fcount, tcount = x
                oof_sum = tsum - fsum
                oof_count = tcount - fcount
                candidates = [(oof_sum + a * folds_prior_map[fold]) / (oof_count + a) for a in self.alphas]
                return candidates

            candidates_pdf_2['_candidates'] = candidates_pdf_2[
                [col_name, self._folds_column, '_fsum', '_tsum', '_fcount', '_tcount']
            ].apply(make_candidates, axis=1)

            scores = []

            def calculate_scores(pd_row):
                folds, target, col, psum, pcount, candidates = pd_row
                score_func = self.score_func_binary if self._task_name == "binary" else self.score_func_reg
                scores.append(
                    [score_func(target, c) * pcount for c in candidates]
                )

            candidates_pdf_3 = _agg.merge(
                candidates_pdf_2[[self._folds_column, col_name, "_candidates"]],
                on=[self._folds_column, col_name]
            )

            candidates_pdf_3.apply(calculate_scores, axis=1)

            _sum = np.array(scores, dtype=np.float64).sum(axis=0)
            _mean = _sum / total_count
            idx = _mean.argmin()

            mapping = {}

            def create_mapping(pd_row):
                folds, col, candidates = pd_row
                mapping[f"{folds}_{col}"] = candidates[idx]

            candidates_pdf_3[
                [self._folds_column, col_name, "_candidates"]
            ].drop_duplicates(subset=[self._folds_column, col_name]).apply(create_mapping, axis=1)

            logger.debug(f"[{type(self)} (TE)] Statistics in pandas have been calculated. Map size: {len(mapping)}")

            # values = sc.broadcast(mapping)
            # cols_to_select.append(te_mapping_udf(values)(_fc, _cur_col).alias(f"{self._fname_prefix}__{col_name}"))

            _column_agg_dicts: dict = candidates_pdf.groupby(by=[col_name]).agg(
                _csum=("_fsum", "sum"), _ccount=("_fcount", "sum")
            ).to_dict()

            self.encodings.append(
                {
                    col_value: (_column_agg_dicts["_csum"][col_value] + self.alphas[idx] * prior)
                               / (_column_agg_dicts["_ccount"][col_value] + self.alphas[idx])
                               for col_value in _column_agg_dicts["_csum"].keys()
                }
            )

            logger.debug(f"[{type(self)} (TE)] Encodings have been calculated")

        logger.info(f"[{type(self)} (TE)] fit_transform is finished")

        return SparkTargetEncoderTransformer(
            encodings=self.encodings,
            input_cols=self.getInputCols(),
            input_roles=self.getInputRoles(),
            output_cols=self.getOutputCols(),
            output_roles=self.getOutputRoles(),
            do_replace_columns=self.getDoReplaceColumns()
        )


class SparkTargetEncoderTransformer(SparkBaseTransformer):

    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self,
                 encodings: List[Dict[str, Any]],
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._encodings = encodings

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        cols_to_select = []
        logger.info(f"[{type(self)} (TE)] transform is started")

        sc = dataset.sql_ctx.sparkSession.sparkContext

        # TODO SPARK-LAMA: Нужно что-то придумать, чтобы ориентироваться по именам колонок, а не их индексу
        # Просто взять и забираться из dataset.features е вариант, т.к. в transform может прийти другой датасет
        # В оригинальной ламе об этом не парились, т.к. сразу переходили в numpy. Если прислали датасет не с тем
        # порядком строк - ну штоош, это проблемы того, кто датасет этот сюда вкинул. Стоит ли нам тоже придерживаться
        # этой логики?
        for i, (col_name, out_name) in enumerate(zip(self.getInputCols(), self.getOutputCols())):
            _cur_col = F.col(col_name)
            logger.debug(f"[{type(self)} (TE)] transform map size for column {col_name}: {len(self._encodings[i])}")

            values = sc.broadcast(self._encodings[i])

            cols_to_select.append(pandas_dict_udf(values)(_cur_col).alias(out_name))

        output = self._make_output_df(dataset, cols_to_select)

        logger.info(f"[{type(self)} (TE)] transform is finished")

        return output


class MultiClassTargetEncoder(ObsoleteSparkTransformer):

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self, alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0)):
        self.alphas = alphas

    #TODO
    @staticmethod
    def score_func(candidates: np.ndarray, target: np.ndarray) -> int:

        target = target[:, np.newaxis, np.newaxis]
        scores = -np.log(np.take_along_axis(candidates, target, axis=1)).mean(axis=0)[0]
        idx = scores.argmin()

        return idx

    def fit_transform(self,
                      dataset: SparkDataset,
                      target_column: str,
                      folds_column: str) -> SparkDataset:

        # set transformer names and add checks
        super().fit(dataset)

        data = dataset.data

        self.encodings = []
        prior = data.groupBy(target_column).agg(F.count(F.col(target_column).cast("double"))/data.count()).collect()
        prior = list(map(lambda x: x[1], prior))

        f_sum = data.groupBy(target_column, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("sumd"))
        f_count = data.groupBy(folds_column).agg(F.count(F.col(target_column).cast("double")).alias("countd"))
        f_sum = f_sum.join(f_count, folds_column)
        tot_sum = data.groupBy(target_column).agg(F.count(F.col(target_column).cast("double")).alias("sumT"))
        f_sum = f_sum.join(tot_sum, target_column)
        tot_count = data.agg(F.count(F.col(target_column).cast("double")).alias("countT")).collect()[0][0]

        folds_prior = f_sum.withColumn("folds_prior", udf(lambda x, y, z: (z - x) / (tot_count - y), FloatType())(F.col("sumd"), F.col("countd"), F.col("sumT")))
        self.feats = data
        self.encs = {}
        self.n_classes = data.agg(F.max(target_column)).collect()[0][0] + 1
        self.old_columns = data.columns
        self.old_columns.remove('_id')
        self._features = []
        for i in dataset.features:
            for j in range(self.n_classes):
                self._features.append("{0}_{1}__{2}".format("multioof", j, i))

        for col_name in self.old_columns:

            f_sum = data.groupBy(col_name, target_column, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("sumd"))
            f_count = data.groupBy(col_name, folds_column).agg(F.count(F.col(target_column).cast("double")).alias("countd"))
            t_sum = data.groupBy(col_name, target_column).agg(F.count(F.col(target_column).cast("double")).alias("sumt"))
            t_count = data.groupBy(col_name).agg(F.count(F.col(target_column).cast("double")).alias("countt"))
            f_sum = f_sum.join(f_count, [col_name, folds_column]).join(t_sum, [col_name, target_column]).join(t_count, col_name)

            oof_sum = f_sum.withColumn("oof_sum", F.col("sumt") - F.col(("sumd"))).withColumn("oof_count", F.col("countt") - F.col(("countd")))
            oof_sum_joined = oof_sum.join(data[col_name, target_column, folds_column], [col_name, target_column, folds_column]).join(folds_prior[folds_column, target_column, "folds_prior"], [folds_column, target_column])
            diff = {}
            udf_diff = udf(lambda x: float(-(np.log(x))), DoubleType())

            for a in self.alphas:
                udf_a = udf(lambda os, fp, oc: (os + a * fp) / (oc + a), DoubleType())
                alp_colname = f"walpha{a}".replace(".", "d").replace(",", "d")
                dif_colname = f"{alp_colname}Diff"
                oof_sum_joined = oof_sum_joined.withColumn(alp_colname, udf_a(F.col("oof_sum").cast("double"), F.col("folds_prior").cast("double"), F.col("oof_count").cast("double")))
                oof_sum_joined = oof_sum_joined.withColumn(dif_colname, udf_diff(F.col(alp_colname).cast("double")))
                diff[a] = oof_sum_joined.agg(F.avg(F.col(alp_colname + "Diff").cast("double"))).collect()[0][0]
            a_opt = min(diff, key=diff.get)
            out_col = f"walpha{a_opt}".replace(".", "d").replace(",", "d")

            w = Window.orderBy(monotonically_increasing_id())

            self.feats = self.feats.withColumn("columnindex", F.row_number().over(w))
            oof_sum_joined = oof_sum_joined.withColumn("columnindex", F.row_number().over(w))

            self.feats = self.feats.alias('a').join(oof_sum_joined.withColumnRenamed(out_col, self._fname_prefix + "__" +col_name).alias('b'),
                                                     self.feats.columnindex == oof_sum_joined.columnindex, 'inner').select(
                [F.col('a.' + xx) for xx in self.feats.columns] + [F.col('b.{}'.format(self._fname_prefix + "__" +col_name))]).drop(self.feats.columnindex)

            # calc best encoding
            enc_list = []
            for i in range(self.n_classes):
                enc = f_sum.withColumn(f"enc_{col_name}", udf(lambda tot_sum, tot_count: float((tot_sum + a_opt * prior[i]) / (tot_count + a_opt)), FloatType())(F.col("sumt"), F.col("countt"))).select(f"enc_{col_name}")
                enc = list(map(lambda x: x[0], enc.collect()))
                enc_list.append(enc)
            sums = [sum(x) for x in zip(*enc_list)]
            for i in range(self.n_classes):
                self.encs[self._fname_prefix + "_" + str(i) + "__" + col_name ] = [enc_list[i][j]/sums[j] for j in range(len(enc_list[i]))]

        self.output_role = NumericRole(np.float32, prob=True)
        return self.transform(dataset)

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        # checks here
        super().transform(dataset)

        data = dataset.data
        dc = self.old_columns
        # transform
        for c in dc:
            data = data.withColumn(c, data[c].cast("int"))
            for i in range(self.n_classes):
                col = self.encs[self._fname_prefix + "_" + str(i) + "__" + c]
                data = data.withColumn(self._fname_prefix + "_" + str(i) + "__" + c, udf(lambda x: col[x], DoubleType())(c))
            data = data.drop(c)

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, self.output_role)

        return output
