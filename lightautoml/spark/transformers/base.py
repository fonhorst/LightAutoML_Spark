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


class SparkSequentialTransformer:
    def __init__(self, transformer_list: List[Union[SparkBaseEstimator, SparkBaseTransformer]]):
        self._transformer_list = copy(transformer_list)

    @property
    def transformers(self) -> List[Union[SparkBaseEstimator, SparkBaseTransformer]]:
        return self._transformer_list


# TODO: SPARK-LAMA make it ABC?
class ObsoleteSparkTransformer(LAMLTransformer):

    _features = []

    _can_unwind_parents: bool = True

    _input_features = []

    def get_output_names(self, input_cols: List[str]) -> List[str]:
        return [f"{self._fname_prefix}__{feat}" for feat in input_cols]

    def fit(self, dataset: SparkDataset, use_features: Optional[List[str]] = None) -> "ObsoleteSparkTransformer":

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

    def _fit(self, dataset: SparkDataset) -> "ObsoleteSparkTransformer":
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


class ColumnsSelectorTransformer(Transformer, HasInputCols):

    def __init__(self,
                 input_cols: Optional[List[str]] = None):
        super().__init__()
        self.set(self.inputCols, input_cols)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return dataset.select(self.getInputCols())


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
