"""Basic classes for features generation."""
import itertools
from copy import copy
from typing import Any, Callable, cast, Dict, Set
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import toposort
from pandas import DataFrame
from pandas import Series
from pyspark.ml import Transformer, Estimator, Pipeline
from pyspark.ml.param import Param, Params
from pyspark.sql import functions as F

from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.pipelines.base import InputFeaturesAndRoles, OutputFeaturesAndRoles
from lightautoml.spark.transformers.base import ChangeRolesTransformer
from lightautoml.spark.transformers.base import SparkBaseEstimator, SparkBaseTransformer, SparkUnionTransformer, \
    SparkSequentialTransformer, SparkEstOrTrans, SparkColumnsAndRoles
from lightautoml.spark.transformers.categorical import SparkCatIntersectionsEstimator, \
    SparkFreqEncoderEstimator, \
    SparkLabelEncoderEstimator, SparkOrdinalEncoderEstimator
from lightautoml.spark.transformers.categorical import SparkTargetEncoderEstimator
from lightautoml.spark.transformers.datetime import SparkBaseDiffTransformer, SparkDateSeasonsTransformer
from lightautoml.spark.transformers.numeric import QuantileBinning


def build_graph(begin: SparkEstOrTrans):
    graph = dict()

    def find_start_end(tr: SparkEstOrTrans) -> Tuple[List[SparkEstOrTrans], List[SparkEstOrTrans]]:
        if isinstance(tr, SparkSequentialTransformer):
            se = [st_or_end for el in tr.transformers for st_or_end in find_start_end(el)]

            starts = se[0]
            ends = se[-1]
            middle = se[1:-1]

            i = 0
            while i < len(middle):
                for new_st, new_end in itertools.product(middle[i], middle[i + 1]):
                    if new_end not in graph:
                        graph[new_end] = set()
                    graph[new_end].add(new_st)
                i += 2

            return starts, ends

        elif isinstance(tr, SparkUnionTransformer):
            se = [find_start_end(el) for el in tr.transformers]
            starts = [s_el for s, _ in se for s_el in s]
            ends = [e_el for _, e in se for e_el in e]
            return starts, ends
        else:
            return [tr], [tr]

    init_starts, _ = find_start_end(begin)

    for st in init_starts:
        if st not in graph:
            graph[st] = set()

    return graph


class SelectTransformer(Transformer):
    colsToSelect = Param(Params._dummy(), "colsToSelect",
                        "columns to select from the dataframe")

    def __init__(self, cols_to_select: List[str]):
        super().__init__()
        self.set(self.colsToSelect, cols_to_select)

    def getColsToSelect(self) -> List[str]:
        return self.getOrDefault(self.colsToSelect)

    def _transform(self, dataset):
        return dataset.select(self.getColsToSelect())


class NoOpTransformer(Transformer):
    def _transform(self, dataset):
        return dataset


class Cacher(Estimator):
    _cacher_dict: Dict[str, SparkDataFrame] = dict()

    @classmethod
    def get_dataset_by_key(cls, key: str) -> Optional[SparkDataFrame]:
        return cls._cacher_dict.get(key, None)

    @property
    def dataset(self) -> SparkDataFrame:
        """Returns chached dataframe"""
        return self._cacher_dict[self._key]

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._dataset: Optional[SparkDataFrame] = None

    def _fit(self, dataset):
        ds = dataset.cache()
        ds.write.mode('overwrite').format('noop').save()

        previous_ds = self._cacher_dict.get(self._key, None)
        if previous_ds:
            previous_ds.unpersist()

        self._cacher_dict[self._key] = ds

        return NoOpTransformer()


class SparkFeaturesPipeline(InputFeaturesAndRoles, OutputFeaturesAndRoles, FeaturesPipeline):
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

    def __init__(self, cacher_key: str = 'default_cacher', **kwargs):
        super().__init__(**kwargs)
        self._cacher_key = cacher_key
        self.pipes: List[Callable[[SparkDataset], SparkEstOrTrans]] = [self.create_pipeline]
        self._transformer: Optional[Transformer] = None

    @property
    def transformer(self) -> Optional[Transformer]:
        return self._transformer

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
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
        assert self.input_features is not None, "Input features should be provided before the fit_transform"
        assert self.input_roles is not None, "Input roles should be provided before the fit_transform"

        pipeline, last_cacher = self._merge(train)

        self._infer_output_features_and_roles(pipeline)

        self._transformer = cast(Transformer, pipeline.fit(train.data))
        sdf = last_cacher.dataset

        features = train.features + self.output_features
        roles = copy(train.roles)
        roles.update(self._output_roles)
        transformed_ds = train.empty()
        transformed_ds.set_data(sdf, features, roles)

        return transformed_ds

    def append(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipes.extend(_pipeline.pipes)

        return self

    def prepend(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in reversed(pipeline):
            self.pipes = _pipeline.pipes + self.pipes

        return self

    def pop(self, i: int = -1) -> Optional[Callable[[SparkDataset], Estimator]]:
        if len(self.pipes) > 1:
            return self.pipes.pop(i)

    def _merge(self, data: SparkDataset) -> Tuple[Estimator, Cacher]:
        est_cachers = [self._optimize_for_caching(pipe(data)) for pipe in self.pipes]
        ests = [e for est, _ in est_cachers for e in est]
        _, last_cacher = est_cachers[-1]
        pipeline = Pipeline(stages=ests)
        return pipeline, last_cacher

    def _optimize_for_caching(self, pipeline: SparkEstOrTrans) -> Tuple[List[Estimator], Cacher]:
        graph = build_graph(pipeline)
        tr_layers = list(toposort.toposort(graph))
        stages = [tr for layer in tr_layers
                  for tr in itertools.chain(layer, [Cacher(self._cacher_key)])]

        last_cacher = stages[-1]
        assert isinstance(last_cacher, Cacher)

        return stages, last_cacher

    def _infer_output_features_and_roles(self, pipeline: Estimator):
        # TODO: infer output features here
        if isinstance(pipeline, Pipeline):
            estimators = pipeline.getStages()
        else:
            estimators = [pipeline]

        assert len(estimators) > 0, "Pipeline cannot be empty"

        fp_input_features = set(self.input_features)

        features = copy(fp_input_features)
        roles = copy(self.input_roles)
        for est in estimators:
            if isinstance(est, Cacher):
                continue

            assert isinstance(est, SparkColumnsAndRoles)

            input_features = est.getInputCols()

            assert not est.getDoReplaceColumns() or all(f not in fp_input_features for f in input_features), \
                "Cannot replace input features of the feature pipeline itself"

            if est.getDoReplaceColumns():
                for col in est.getInputCols():
                    features.remove(col)
                    del roles[col]

            assert not any(f in features for f in est.getOutputCols()), \
                "Cannot add an already existing feature"

            features.update(est.getOutputCols())
            roles.update(est.getOutputRoles())

        assert all((f in features) for f in fp_input_features), \
            "All input features should be present in the output features"

        assert all((f in roles) for f in fp_input_features), \
            "All input features should be present in the output roles"

        # we want to have only newly added features in out output features, not input features
        for col in fp_input_features:
            features.remove(col)
            del roles[col]

        self._output_roles = roles

    def release_cache(self):
        sdf = Cacher.get_dataset_by_key(self._cacher_key)
        if sdf is not None:
            sdf.unpersist()


class SparkTabularDataFeatures:
    """Helper class contains basic features transformations for tabular data.

    This method can de shared by all tabular feature pipelines,
    to simplify ``.create_automl`` definition.
    """

    def __init__(self, **kwargs: Any):
        """Set default parameters for tabular pipeline constructor.

        Args:
            **kwargs: Additional parameters.

        """
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

    def _get_input_features(self) -> Set[str]:
        raise NotImplementedError()

    def _cols_by_role(self, dataset: SparkDataset, role_name: str, **kwargs: Any) -> List[str]:
        cols = get_columns_by_role(dataset, role_name, **kwargs)
        filtered_cols = [col for col in cols if col in self._get_input_features()]
        return filtered_cols

    def get_cols_for_datetime(self, train: SparkDataset) -> Tuple[List[str], List[str]]:
        """Get datetime columns to calculate features.

        Args:
            train: Dataset with train data.

        Returns:
            2 list of features names - base dates and common dates.

        """
        base_dates = self._cols_by_role(train, "Datetime", base_date=True)
        datetimes = self._cols_by_role(train, "Datetime", base_date=False) + self._cols_by_role(
            train, "Datetime", base_date=True, base_feats=True
        )

        return base_dates, datetimes

    def get_datetime_diffs(self, train: SparkDataset) -> Optional[SparkBaseTransformer]:
        """Difference for all datetimes with base date.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer or ``None`` if no required features.

        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return None

        roles = {f: train.roles[f] for f in itertools.chain(base_dates, datetimes)}

        base_diff = SparkBaseDiffTransformer(
            input_roles=roles,
            base_names=base_dates,
            diff_names=datetimes
        )

        return base_diff

    def get_datetime_seasons(
        self, train: SparkDataset, outp_role: Optional[ColumnRole] = None
    ) -> Optional[SparkBaseTransformer]:
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

        roles = {f: train.roles[f] for f in datetimes}

        date_as_cat = SparkDateSeasonsTransformer(input_cols=datetimes, input_roles=roles, output_role=outp_role)

        return date_as_cat

    def get_numeric_data(
        self,
        train: SparkDataset,
        feats_to_select: Optional[List[str]] = None,
        prob: Optional[bool] = None,
    ) -> Optional[SparkBaseTransformer]:
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
                feats_to_select = self._cols_by_role(train, "Numeric")
            else:
                feats_to_select = self._cols_by_role(train, "Numeric", prob=prob)

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        num_processing = ChangeRolesTransformer(input_cols=feats_to_select,
                                                input_roles=roles,
                                                role=NumericRole(np.float32))

        return num_processing

    def get_freq_encoding(self, train: SparkDataset, feats_to_select: Optional[List[str]] = None) \
            -> Optional[SparkBaseEstimator]:
        """Get frequency encoding part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", encoding_type="freq")

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkFreqEncoderEstimator(input_cols=feats_to_select, input_roles=roles)

        return cat_processing

    def get_ordinal_encoding(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get order encoded part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", ordinal=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        ord = SparkOrdinalEncoderEstimator(input_cols=feats_to_select,
                                           input_roles=roles,
                                           subs=self.subsample,
                                           random_state=self.random_state)

        return ord

    def get_categorical_raw(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
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
                feats = self._cols_by_role(train, "Category", encoding_type=i)
                feats_to_select.extend(feats)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkLabelEncoderEstimator(input_cols=feats_to_select,
                                                    input_roles=roles,
                                                    subs=self.subsample,
                                                    random_state=self.random_state)
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
                target_encoder = SparkTargetEncoderEstimator
            else:
                tds = cast(SparkDataFrame, train.target)
                result = tds.select(F.max(train.target_column).alias("max")).first()
                n_classes = result['max'] + 1

                # TODO: SPARK-LAMA add warning here
                target_encoder = None
                raise NotImplementedError()
                # if n_classes <= self.multiclass_te_co:
                #     target_encoder = MultiClassTargetEncoder

        return target_encoder

    def get_binned_data(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseTransformer]:
        """Get encoded quantiles of numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: features to hanlde. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Numeric", discretization=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        binned_processing = QuantileBinning(nbins=self.max_bin_count)

        return binned_processing

    def get_categorical_intersections(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
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

        roles = {f: train.roles[f] for f in feats_to_select}

        # TODO: removed from CatIntersection
        # subs = self.subsample,
        # random_state = self.random_state,

        cat_processing = SparkCatIntersectionsEstimator(input_cols=feats_to_select,
                                                        input_roles=roles,
                                                        max_depth=self.max_intersection_depth)

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
