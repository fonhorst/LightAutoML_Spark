"""Base class for selection pipelines."""
from abc import ABC
from copy import copy
from typing import Optional, List, cast

from lightautoml.dataset.base import RolesDict
from lightautoml.pipelines.selection.base import SelectionPipeline, EmptySelector, ImportanceEstimator
from lightautoml.validation.base import TrainValidIterator
from pandas import Series
from pyspark.ml import Transformer

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.transformers.base import ColumnsSelectorTransformer
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, SparkSelectionPipeline


class SparkImportanceEstimator(ImportanceEstimator, ABC):
    def __init__(self):
        super(SparkImportanceEstimator, self).__init__()


class SparkSelectionPipelineWrapper(SparkSelectionPipeline, TransformerInputOutputRoles):
    def __init__(self, sel_pipe: SelectionPipeline):
        assert not sel_pipe.is_fitted, "Cannot work with prefitted SelectionPipeline"
        assert isinstance(sel_pipe.features_pipeline, SparkFeaturesPipeline) or isinstance(sel_pipe, EmptySelector), \
            "SelectionPipeline should have SparkFeaturePipeline as features_pipeline"
        self._sel_pipe = sel_pipe
        self._service_columns = None
        self._is_fitted = False
        self._input_roles: Optional[RolesDict] = None
        self._output_roles: Optional[RolesDict] = None
        self._feature_pipeline = cast(SparkFeaturesPipeline, self._sel_pipe.features_pipeline)
        super().__init__()

    @property
    def transformer(self, *args, **kwargs) -> Optional[Transformer]:
        if not self._sel_pipe.is_fitted:
            return None

        return ColumnsSelectorTransformer(
            input_cols=[*self._service_columns, *self._sel_pipe.selected_features]
        )

    @property
    def input_roles(self) -> Optional[RolesDict]:
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        return self._output_roles

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def selected_features(self) -> List[str]:
        return self._sel_pipe.selected_features

    @property
    def in_features(self) -> List[str]:
        return self._sel_pipe.in_features

    @property
    def dropped_features(self) -> List[str]:
        return self._sel_pipe.dropped_features

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        self._sel_pipe.perform_selection(train_valid)

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        self._service_columns = train_valid.train.service_columns
        self._sel_pipe.fit(train_valid)
        self._is_fitted = True

        self._input_roles = copy(train_valid.train.roles)
        self._output_roles = {
            feat: role
            for feat, role in self._input_roles.items()
            if feat in self._sel_pipe.selected_features
        }

    def select(self, dataset: SparkDataset) -> SparkDataset:
        return cast(SparkDataset, self._sel_pipe.select(dataset))

    def map_raw_feature_importances(self, raw_importances: Series):
        return self._sel_pipe.map_raw_feature_importances(raw_importances)

    def get_features_score(self):
        return self._sel_pipe.get_features_score()
