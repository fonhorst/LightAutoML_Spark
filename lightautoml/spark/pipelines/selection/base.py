"""Base class for selection pipelines."""
from abc import ABC
from typing import Any, Optional, Union, Tuple, List

from pyspark.ml.param.shared import HasInputCols, HasOutputCols

from lightautoml.dataset.base import LAMLDataset, RolesDict
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.pipelines.features.base import FeaturesPipeline

from lightautoml.pipelines.selection.base import SelectionPipeline, ImportanceEstimator, EmptySelector
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.pipelines.features.base import SparkFeaturesPipeline
from lightautoml.spark.transformers.base import HasInputRoles, HasOutputRoles
from lightautoml.validation.base import TrainValidIterator


class SparkImportanceEstimator:
    """
    Abstract class, that estimates feature importances.
    """

    def __init__(self):
        self.raw_importances = None

    # Change signature here to be compatible with MLAlgo
    def fit(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def get_features_score(self) -> SparkDataset:

        return self.raw_importances


class SparkSelectionPipeline(SelectionPipeline, ABC):
    def __init__(self,
                 input_features: List[str],
                 input_roles: RolesDict,
                 features_pipeline: Optional[SparkFeaturesPipeline] = None,
                 ml_algo: Optional[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]] = None,
                 imp_estimator: Optional[ImportanceEstimator] = None,
                 fit_on_holdout: bool = False,
                 **kwargs: Any):
        super().__init__(features_pipeline, ml_algo, imp_estimator, fit_on_holdout, **kwargs)
        self._input_features = input_features
        self._input_roles = input_roles

    @property
    def input_features(self) -> List[str]:
        return self._input_features

    @property
    def input_roles(self) -> RolesDict:
        return self._input_roles

    def select(self, dataset: LAMLDataset):
        raise NotImplementedError("Not supported for Spark version")
        pass


class SparkEmptySelector(SparkSelectionPipeline):
    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        self._selected_features = self.input_features


class SparkImportanceCutoffSelector(SparkSelectionPipeline, ImportanceCutoffSelector):
    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict,
                 features_pipeline: Optional[SparkFeaturesPipeline] = None,
                 ml_algo: Optional[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]] = None,
                 imp_estimator: Optional[ImportanceEstimator] = None,
                 fit_on_holdout: bool = False,
                 cutoff: float = 0.0):
        super().__init__(input_cols, input_roles, features_pipeline, ml_algo, imp_estimator, fit_on_holdout)
        self._cutoff = cutoff

    def perform_selection(self, train_valid: Optional[TrainValidIterator] = None):
        """Select features based on cutoff value.

        Args:
            train_valid: Not used.

        """
        imp = self.imp_estimator.get_features_score()
        self.map_raw_feature_importances(imp)
        selected = self.mapped_importances.index.values[self.mapped_importances.values > self.cutoff]
        if len(selected) == 0:
            selected = self.mapped_importances.index.values[:1]
        self._selected_features = list(selected)
