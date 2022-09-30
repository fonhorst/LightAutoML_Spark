"""Base class for selection pipelines."""
from typing import Any, Optional, List, cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.validation.base import TrainValidIterator
from pandas import Series
from pyspark.ml import Transformer

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.transformers.base import ColumnsSelectorTransformer


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


class SparkSelectionPipelineWrapper(SelectionPipeline):
    def __init__(self, sel_pipe: SelectionPipeline):
        assert not sel_pipe.is_fitted, "Cannot work with prefitted SelectionPipeline"
        self._sel_pipe = sel_pipe
        self._service_columns = None
        self._is_fitted = False
        super().__init__()

    @property
    def transformer(self) -> Optional[Transformer]:
        if not self._sel_pipe.is_fitted:
            return None

        return ColumnsSelectorTransformer(
            input_cols=[self._service_columns, *self._sel_pipe.selected_features]
        )

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

    def fit(self, train_valid: TrainValidIterator):
        self._service_columns = cast(SparkDataset, train_valid.train).service_columns
        self._sel_pipe.fit(train_valid)
        self._is_fitted = True

    def select(self, dataset: SparkDataset) -> SparkDataset:
        return cast(SparkDataset, self._sel_pipe.select(dataset))

    def map_raw_feature_importances(self, raw_importances: Series):
        return self._sel_pipe.map_raw_feature_importances(raw_importances)

    def get_features_score(self):
        return self._sel_pipe.get_features_score()


