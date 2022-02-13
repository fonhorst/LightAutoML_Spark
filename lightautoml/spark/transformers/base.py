import logging
from abc import ABC
from copy import copy, deepcopy
from typing import Dict, cast, Sequence, List, Set, Optional, Union

from pyspark.sql import Column

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.utils import concatenate
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.mlwriters import TmpСommonMLWriter
from lightautoml.spark.utils import log_exec_time
from lightautoml.transformers.base import LAMLTransformer, ColumnsSelector as LAMAColumnsSelector, \
    ChangeRoles as LAMAChangeRoles
from lightautoml.transformers.base import Roles

from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, TypeConverters
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import MLReadable, MLWritable, MLWriter


logger = logging.getLogger(__name__)


SparkEstOrTrans = Union[
    'SparkBaseEstimator',
    'SparkBaseTransformer',
    'SparkUnionTransformer',
    'SparkSequentialTransformer'
]


class HasInputRoles(Params):
    """
    Mixin for param inputCols: input column names.
    """

    inputRoles = Param(Params._dummy(), "inputRoles",
                       "input roles (lama format)")

    def __init__(self):
        super().__init__()

    def getInputRoles(self):
        """
        Gets the value of inputCols or its default value.
        """
        return self.getOrDefault(self.inputRoles)


class HasOutputRoles(Params):
    """
    Mixin for param inputCols: input column names.
    """
    outputRoles = Param(Params._dummy(), "outputRoles",
                        "output roles (lama format)")

    def __init__(self):
        super().__init__()

    def getOutputRoles(self):
        """
        Gets the value of inputCols or its default value.
        """
        return self.getOrDefault(self.outputRoles)


class SparkColumnsAndRoles(HasInputCols, HasOutputCols, HasInputRoles, HasOutputRoles):
    doReplaceColumns = Param(Params._dummy(), "doReplaceColumns", "whatever it replaces columns or not")

    def getDoReplaceColumns(self):
        return self.getOrDefault(self.doReplaceColumns)

    @staticmethod
    def make_dataset(transformer: 'SparkColumnsAndRoles', base_dataset: SparkDataset, data: SparkDataFrame) -> SparkDataset:
        # TODO: SPARK-LAMA deepcopy?
        new_roles = copy(base_dataset.roles)
        new_roles.update(transformer.getOutputRoles())
        new_ds = base_dataset.empty()
        new_ds.set_data(data, base_dataset.features + transformer.getOutputCols(),  new_roles)
        return new_ds


class SparkBaseEstimator(Estimator, SparkColumnsAndRoles, MLWritable, ABC):
    _fit_checks = ()
    _fname_prefix = ""

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 do_replace_columns: bool = False,
                 output_role: Optional[ColumnRole] = None):
        super().__init__()

        self._output_role = output_role

        assert all((f in input_roles) for f in input_cols), \
            "All columns should have roles"

        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, self._make_output_names(input_cols))
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, self._make_output_roles())
        self.set(self.doReplaceColumns, do_replace_columns)

    def _make_output_names(self, input_cols: List[str]) -> List[str]:
        return [f"{self._fname_prefix}__{feat}" for feat in input_cols]

    def _make_output_roles(self):
        assert self._output_role is not None
        new_roles = {}
        new_roles.update({feat: self._output_role for feat in self.getOutputCols()})
        return new_roles

    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Estimator instance."
        return TmpСommonMLWriter(self.uid)


class SparkBaseTransformer(Transformer, SparkColumnsAndRoles, MLWritable, ABC):
    _fname_prefix = ""

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__()

        # assert len(input_cols) == len(output_cols)
        # assert len(input_roles) == len(output_roles)
        assert all((f in input_roles) for f in input_cols)
        assert all((f in output_roles) for f in output_cols)

        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, output_cols)
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, output_roles)
        self.set(self.doReplaceColumns, do_replace_columns)

    _transform_checks = ()

    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return TmpСommonMLWriter(self.uid)

    def _make_output_df(self, input_df: SparkDataFrame, cols_to_add: List[Union[str, Column]]):
        if not self.getDoReplaceColumns():
            return input_df.select('*', *cols_to_add)

        input_cols = set(self.getInputCols())
        cols_to_leave = [f for f in input_df.columns if f not in input_cols]
        return input_df.select(*cols_to_leave, *cols_to_add)


class SparkUnionTransformer:
    def __init__(self, transformer_list: List[Union[SparkBaseEstimator, SparkBaseTransformer]]):
        self._transformer_list = copy(transformer_list)

    @property
    def transformers(self) -> List[Union[SparkBaseEstimator, SparkBaseTransformer]]:
        return self._transformer_list

    def _find_last_stage(self, stage):
        if isinstance(stage, SparkSequentialTransformer):
            stage = stage.transformers[-1]
            return stage
        return stage

    def get_output_cols(self) -> List[str]:
        """Get list of output columns from all stages

        Returns:
            List[str]: output columns
        """
        output_cols = []
        for stage in self._transformer_list:
            stage = self._find_last_stage(stage)
            output_cols.extend(stage.getOutputCols())
        return list(set(output_cols))

    def get_output_roles(self) -> RolesDict:
        """Get output roles from all stages

        Returns:
            RolesDict: output roles
        """
        roles = {}
        for stage in self._transformer_list:
            stage = self._find_last_stage(stage)
            roles.update(deepcopy(stage.getOutputRoles()))        

        return roles


class SparkSequentialTransformer:
    def __init__(self, transformer_list: List[Union[SparkBaseEstimator, SparkBaseTransformer]]):
        self._transformer_list = copy(transformer_list)

    @property
    def transformers(self) -> List[Union[SparkBaseEstimator, SparkBaseTransformer]]:
        return self._transformer_list


# TODO: SPARK-LAMA make it ABC?
class SparkTransformer(LAMLTransformer):

    _features = []

    _can_unwind_parents: bool = True

    _input_features = []

    def get_output_names(self, input_cols: List[str]) -> List[str]:
        return [f"{self._fname_prefix}__{feat}" for feat in input_cols]

    def fit(self, dataset: SparkDataset, use_features: Optional[List[str]] = None) -> "SparkTransformer":

        logger.info(f"SparkTransformer of type: {type(self)}")

        if use_features:
            existing_feats = set(dataset.features)
            not_found_feats = [feat for feat in use_features if feat not in existing_feats]
            assert len(not_found_feats) == 0, \
                f"Not found features {not_found_feats} among existing {existing_feats}"
            self._features = use_features

            # # TODO: SPARK-LAMA reimplement it later
            # # here we intentionally is going to reduce features to the desired
            # # to pass the check with rewriting checks themselves
            # use_roles = {feat: dataset.roles[feat] for feat in use_features}
            # ds = dataset.empty()
            # ds.set_data(dataset.data, use_features, use_roles, dataset.dependencies)

            # for check_func in self._fit_checks:
            #     check_func(ds)
        else:
            self._features = dataset.features

        self._input_features = use_features if use_features else dataset.features

        for check_func in self._fit_checks:
            check_func(dataset, use_features)

        return self._fit(dataset)

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        for check_func in self._transform_checks:
            check_func(dataset)

        return self._transform(dataset)

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        # TODO: SPARK-LAMA probably we should assume
        #  that fit_transform executes with cache by default
        #  e.g fit_transform returns a cached and materialized dataset
        logger.info(f"fit_transform in {self._fname_prefix}: {type(self)}")

        dataset.cache()
        self.fit(dataset)

        # when True, it means that during fit operation we conducted some action that
        # materialized our current dataset and thus we can unpersist all its dependencies
        # because we have data to propagate in the cache already
        if self._can_unwind_parents:
            dataset.unwind_dependencies()
            deps = [dataset]
        else:
            deps = dataset.dependencies

        result = self.transform(dataset)
        result.dependencies = deps

        return result

    @staticmethod
    def _get_updated_roles(dataset: SparkDataset, new_features: List[str], new_role: ColumnRole) -> RolesDict:
        new_roles = deepcopy(dataset.roles)
        new_roles.update({feat: new_role for feat in new_features})
        return new_roles


class SequentialTransformer(SparkTransformer):
    """
    Transformer that contains the list of transformers and apply one by one sequentially.
    """
    _fname_prefix = "seq"

    def __init__(self, transformer_list: Sequence[SparkTransformer], is_already_fitted: bool = False):
        """

        Args:
            transformer_list: Sequence of transformers.

        """
        self.transformer_list = transformer_list
        self._is_fitted = is_already_fitted

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        """Fit not supported. Needs output to fit next transformer.

        Args:
            dataset: Dataset to fit.

        """
        raise NotImplementedError("Sequential supports only fit_transform.")

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Apply the sequence of transformers to dataset one over output of previous.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        logger.info(f"[{type(self)} (SEQ)] transform is started")
        for trf in self.transformer_list:
            dataset = trf.transform(dataset)

        logger.info(f"[{type(self)} (SEQ)] transform is finished")

        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Sequential ``.fit_transform``.

         Output features - features from last transformer with no prefix.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        logger.info(f"[{type(self)} (SEQ)] fit_transform is started")
        if not self._is_fitted:
            for trf in self.transformer_list:
                dataset = trf.fit_transform(dataset)
        else:
            dataset = self.transform(dataset)

        self.features = self.transformer_list[-1].features
        logger.info(f"[{type(self)} (SEQ)] fit_transform is finished")
        return dataset


class UnionTransformer(SparkTransformer):
    """Transformer that apply the sequence on transformers in parallel on dataset and concatenate the result."""

    _fname_prefix = "union"

    def __init__(self, transformer_list: Sequence[SparkTransformer], n_jobs: int = 1):
        """

        Args:
            transformer_list: Sequence of transformers.
            n_jobs: Number of processes to run fit and transform.

        """
        # TODO: Add multiprocessing version here
        self.transformer_list = [x for x in transformer_list if x is not None]
        self.n_jobs = n_jobs

        assert len(self.transformer_list) > 0, "The list of transformers cannot be empty or contains only None-s"

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        fnames = []
        logger.info(f"[{type(self)} (UNI)] fit is started")
        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                trf.fit(dataset)
                fnames.append(trf.features)

        self.features = fnames
        logger.info(f"[{type(self)} (UNI)] fit is finished")
        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        res = []
        logger.info(f"[{type(self)} (UNI)] transform is started")
        for trf in self.transformer_list:
            ds = trf.transform(dataset)
            res.append(ds)

        union_res = SparkDataset.concatenate(res)
        logger.info(f"[{type(self)} (UNI)] transform is finished")
        return union_res

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Fit and transform transformers in parallel.
         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit and transform on.

        Returns:
            Dataset with new features.

        """
        res = []
        actual_transformers = []
        logger.info(f"[{type(self)} (UNI)] fit_transform is started")

        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                ds = trf.fit_transform(dataset)
                # if ds:
                res.append(ds)
                actual_transformers.append(trf)

        # this concatenate operations also propagates all dependencies
        logger.info(f"[{type(self)} (UNI)] fit_transform: concat is started")
        result = SparkDataset.concatenate(res) if len(res) > 0 else None
        logger.info(f"[{type(self)} (UNI)] fit_transform: concat is finished")

        self.transformer_list = actual_transformers
        self.features = result.features
        logger.info(f"[{type(self)} (UNI)] fit_transform is finished")
        return result


class ColumnsSelector(LAMAColumnsSelector, SparkTransformer):
    _fname_prefix = "colsel"
    _can_unwind_parents = False


class ColumnsSelectorTransformer(Transformer, HasInputCols):

    def __init__(self,
                 input_cols: Optional[List[str]] = None):
        super().__init__()
        self.set(self.inputCols, input_cols)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return dataset.select(self.getInputCols())


class ChangeRoles(LAMAChangeRoles, SparkTransformer):
    _fname_prefix = "changeroles"
    _can_unwind_parents = False

    def get_output_names(self, input_cols: List[str]) -> List[str]:
        return copy(input_cols)


class ChangeRolesTransformer(SparkBaseTransformer):
    # Note: this trasnformer cannot be applied directly to input columns of a feature pipeline
    def __init__(self, 
                 input_cols: List[str],
                 input_roles: RolesDict,
                 role: ColumnRole):
        super().__init__(
            input_cols=input_cols,
            output_cols=input_cols,
            input_roles=input_roles,
            output_roles={f: deepcopy(role) for f in input_cols},
            do_replace_columns=True)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return dataset
