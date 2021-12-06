"""Basic classes for features generation."""

from copy import copy, deepcopy
from typing import Any, Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from pandas import DataFrame
from pandas import Series
from pyspark.sql import functions as F

from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.categorical import FreqEncoder, OrdinalEncoder, LabelEncoder, \
    TargetEncoder, MultiClassTargetEncoder, CatIntersectstions
from lightautoml.spark.transformers.datetime import BaseDiff, DateSeasons
from lightautoml.spark.transformers.base import SequentialTransformer, ColumnsSelector, ChangeRoles, \
    UnionTransformer, SparkTransformer
from lightautoml.pipelines.utils import map_pipeline_names
from lightautoml.spark.transformers.numeric import QuantileBinning


class FeaturesPipeline:
    """Abstract class.

    Analyze train dataset and create composite transformer
    based on subset of features.
    Instance can be interpreted like Transformer
    (look for :class:`~lightautoml.transformers.base.LAMLTransformer`)
    with delayed initialization (based on dataset metadata)
    Main method, user should define in custom pipeline is ``.create_pipeline``.
    For example, look at
    :class:`~lightautoml.pipelines.features.lgb_pipeline.LGBSimpleFeatures`.
    After FeaturePipeline instance is created, it is used like transformer
    with ``.fit_transform`` and ``.transform`` method.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipes: List[Callable[[SparkDataset], SparkTransformer]] = [self.create_pipeline]
        self.sequential = False

    # TODO: visualize pipeline ?
    @property
    def input_features(self) -> List[str]:
        """Names of input features of train data."""
        return self._input_features

    @input_features.setter
    def input_features(self, val: List[str]):
        """Setter for input_features.

        Args:
            val: List of strings.

        """
        self._input_features = deepcopy(val)

    @property
    def output_features(self) -> List[str]:
        """List of feature names that produces _pipeline."""
        return self._pipeline.features

    @property
    def used_features(self) -> List[str]:
        """List of feature names from original dataset that was used to produce output."""
        mapped = map_pipeline_names(self.input_features, self.output_features)
        return list(set(mapped))

    def create_pipeline(self, train: SparkDataset) -> SparkTransformer:
        """Analyse dataset and create composite transformer.

        Args:
            train: Dataset with train data.

        Returns:
            Composite transformer (pipeline).

        """
        raise NotImplementedError

    def fit_transform(self, train: SparkDataset) -> SparkDataset:
        """Create pipeline and then fit on train data and then transform.

        Args:
            train: Dataset with train data.n

        Returns:
            Dataset with new features.

        """
        # TODO: Think about input/output features attributes
        self._input_features = train.features
        self._pipeline = self._merge_seq(train) if self.sequential else self._merge(train)

        # TODO: LAMA-SPARK a place with potential duplicate computations
        #        need to think carefully about it

        return self._pipeline.fit_transform(train)

    def transform(self, test: SparkDataset) -> SparkDataset:
        """Apply created pipeline to new data.

        Args:
            test: Dataset with test data.

        Returns:
            Dataset with new features.

        """
        return self._pipeline.transform(test)

    def set_sequential(self, val: bool = True):
        self.sequential = val
        return self

    def append(self, pipeline):
        if isinstance(pipeline, FeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipes.extend(_pipeline.pipes)

        return self

    def prepend(self, pipeline):
        if isinstance(pipeline, FeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in reversed(pipeline):
            self.pipes = _pipeline.pipes + self.pipes

        return self

    def pop(self, i: int = -1) -> Optional[Callable[[SparkDataset], SparkTransformer]]:
        if len(self.pipes) > 1:
            return self.pipes.pop(i)

    def _merge(self, data: SparkDataset) -> SparkTransformer:
        pipes = []
        for pipe in self.pipes:
            pipes.append(pipe(data))

        return UnionTransformer(pipes) if len(pipes) > 1 else pipes[-1]

    def _merge_seq(self, data: SparkDataset) -> SparkTransformer:
        pipes = []
        for pipe in self.pipes:
            _pipe = pipe(data)
            data = _pipe.fit_transform(data)
            pipes.append(_pipe)

        return SequentialTransformer(pipes) if len(pipes) > 1 else pipes[-1]


# The class is almost identical to what we have in regular LAMA
# But we cannot reuse it directly due to:
# - imports of transformers directly from lightautoml module
#   (probably, can be solved through conditional imports in transformer's modules __init__ file,
#   but everyone should use imports from __init__ instead of direct imports)
# - NumpyPandas dataset used in some places, including ConvertDataset
#   (can be solved via ConvertDataset replacement with SparkDataset that do nothing,
#   NumpyPandas should be replaced with an appropriate base class)
# - self.get_uniques_cnt - this methods works with data,
#   not metadata and thus requires rewriting
#   (can be replace with an external function that can be substituted)
class TabularDataFeatures:
    """Helper class contains basic features transformations for tabular data.

    This method can de shared by all tabular feature pipelines,
    to simplify ``.create_automl`` definition.
    """

    def __init__(self, **kwargs: Any):
        """Set default parameters for tabular pipeline constructor.

        Args:
            **kwargs: Additional parameters.

        """
        print("spark tdf ctr")
        self.multiclass_te_co = 3
        self.top_intersections = 5
        self.max_intersection_depth = 3
        self.subsample = 0.1 #10000
        self.random_state = 42
        self.feats_imp = None
        self.ascending_by_cardinality = False

        self.max_bin_count = 10
        self.sparse_ohe = "auto"

        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    @staticmethod
    def get_cols_for_datetime(train: SparkDataset) -> Tuple[List[str], List[str]]:
        """Get datetime columns to calculate features.

        Args:
            train: Dataset with train data.

        Returns:
            2 list of features names - base dates and common dates.

        """
        base_dates = get_columns_by_role(train, "Datetime", base_date=True)
        datetimes = get_columns_by_role(train, "Datetime", base_date=False) + get_columns_by_role(
            train, "Datetime", base_date=True, base_feats=True
        )

        return base_dates, datetimes

    def get_datetime_diffs(self, train: SparkDataset) -> Optional[SparkTransformer]:
        """Difference for all datetimes with base date.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer or ``None`` if no required features.

        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return

        dt_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=list(set(datetimes + base_dates))),
                BaseDiff(base_names=base_dates, diff_names=datetimes),
            ]
        )
        return dt_processing

    def get_datetime_seasons(
        self, train: SparkDataset, outp_role: Optional[ColumnRole] = None
    ) -> Optional[SparkTransformer]:
        """Get season params from dates.

        Args:
            train: Dataset with train data.
            outp_role: Role associated with output features.

        Returns:
            Transformer or ``None`` if no required features.

        """
        _, datetimes = self.get_cols_for_datetime(train)
        for col in copy(datetimes):
            if len(train.roles[col].seasonality) == 0 and train.roles[col].country is None:
                datetimes.remove(col)

        if len(datetimes) == 0:
            return

        if outp_role is None:
            outp_role = NumericRole(np.float32)

        date_as_cat = SequentialTransformer(
            [
                ColumnsSelector(keys=datetimes),
                DateSeasons(outp_role),
            ]
        )
        return date_as_cat

    @staticmethod
    def get_numeric_data(
        train: SparkDataset,
        feats_to_select: Optional[List[str]] = None,
        prob: Optional[bool] = None,
    ) -> Optional[SparkTransformer]:
        """Select numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.
            prob: Probability flag.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            if prob is None:
                feats_to_select = get_columns_by_role(train, "Numeric")
            else:
                feats_to_select = get_columns_by_role(train, "Numeric", prob=prob)

        if len(feats_to_select) == 0:
            return

        num_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                # we don't need this because everything is handled by Spark
                # thus we have no other dataset type except SparkDataset
                # ConvertDataset(dataset_type=NumpyDataset),
                ChangeRoles(NumericRole(np.float32)),
            ]
        )

        return num_processing

    @staticmethod
    def get_freq_encoding(
        train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkTransformer]:
        """Get frequency encoding part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, "Category", encoding_type="freq")

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                FreqEncoder(),
            ]
        )
        return cat_processing

    def get_ordinal_encoding(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkTransformer]:
        """Get order encoded part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, "Category", ordinal=True)

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                OrdinalEncoder(subs=self.subsample, random_state=self.random_state),
            ]
        )
        return cat_processing

    def get_categorical_raw(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkTransformer]:
        """Get label encoded categories data.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:
            feats_to_select = []
            for i in ["auto", "oof", "int", "ohe"]:
                feats_to_select.extend(get_columns_by_role(train, "Category", encoding_type=i))

        if len(feats_to_select) == 0:
            return

        cat_processing = [
            ColumnsSelector(keys=feats_to_select),
            LabelEncoder(subs=self.subsample, random_state=self.random_state),
        ]
        cat_processing = SequentialTransformer(cat_processing)
        return cat_processing

    def get_target_encoder(self, train: SparkDataset) -> Optional[type]:
        """Get target encoder func for dataset.

        Args:
            train: Dataset with train data.

        Returns:
            Class

        """
        target_encoder = None
        if train.folds is not None:
            if train.task.name in ["binary", "reg"]:
                target_encoder = TargetEncoder
            else:
                n_classes = train.target.max() + 1
                if n_classes <= self.multiclass_te_co:
                    target_encoder = MultiClassTargetEncoder

        return target_encoder

    def get_binned_data(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkTransformer]:
        """Get encoded quantiles of numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: features to hanlde. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, "Numeric", discretization=True)

        if len(feats_to_select) == 0:
            return

        binned_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                QuantileBinning(nbins=self.max_bin_count),
            ]
        )
        return binned_processing

    def get_categorical_intersections(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkTransformer]:
        """Get transformer that implements categorical intersections.

        Args:
            train: Dataset with train data.
            feats_to_select: features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:

            categories = get_columns_by_role(train, "Category")
            feats_to_select = categories

            if len(categories) <= 1:
                return

            elif len(categories) > self.top_intersections:
                feats_to_select = self.get_top_categories(train, self.top_intersections)

        elif len(feats_to_select) <= 1:
            return

        # TODO: removed from CatIntersection
        # subs = self.subsample,
        # random_state = self.random_state,
        cat_processing = [
            ColumnsSelector(keys=feats_to_select),
            CatIntersectstions(
                # intersections=feats_to_select,
                max_depth=self.max_intersection_depth
            ),
        ]
        cat_processing = SequentialTransformer(cat_processing)

        return cat_processing

    def get_uniques_cnt(self, train: SparkDataset, feats: List[str]) -> Series:
        """Get unique values cnt.

        Be aware that this function uses approx_count_distinct and thus cannot return precise results

        Args:
            train: Dataset with train data.
            feats: Features names.

        Returns:
            Series.

        """

        # TODO: LAMA-SPARK: should be conditioned on a global setting
        #       producing either an error or warning
        # assert not train.data.is_cached, "The train dataset should be cached before executing this operation"

        sdf = train.data.select(feats)

        # TODO SPARK-LAMA: Do we really need this sampling?
        # if self.subsample:
        #     sdf = sdf.sample(withReplacement=False, fraction=self.subsample, seed=self.random_state)

        sdf = sdf.select([F.approx_count_distinct(col).alias(col) for col in feats])
        result = sdf.collect()[0]

        uns = [result[col] for col in feats]
        return Series(uns, index=feats, dtype="int")

    def get_top_categories(self, train: SparkDataset, top_n: int = 5) -> List[str]:
        """Get top categories by importance.

        If feature importance is not defined,
        or feats has same importance - sort it by unique values counts.
        In second case init param ``ascending_by_cardinality``
        defines how - asc or desc.

        Args:
            train: Dataset with train data.
            top_n: Number of top categories.

        Returns:
            List.

        """
        if self.max_intersection_depth <= 1 or self.top_intersections <= 1:
            return []

        cats = get_columns_by_role(train, "Category")
        if len(cats) == 0:
            return []

        df = DataFrame({"importance": 0, "cardinality": 0}, index=cats)
        # importance if defined
        if self.feats_imp is not None:
            feats_imp = Series(self.feats_imp.get_features_score()).sort_values(ascending=False)
            df["importance"] = feats_imp[feats_imp.index.isin(cats)]
            df["importance"].fillna(-np.inf)

        # check for cardinality
        df["cardinality"] = self.get_uniques_cnt(train, cats)
        # sort
        df = df.sort_values(
            by=["importance", "cardinality"],
            ascending=[False, self.ascending_by_cardinality],
        )
        # get top n
        top = list(df.index[:top_n])

        return top
