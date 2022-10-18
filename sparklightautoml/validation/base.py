from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import copy
from typing import Tuple, cast, Sequence, Optional, Union, Any

from lightautoml.ml_algo.base import MLAlgo
from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.selection.base import SelectionPipeline, ImportanceEstimator
from lightautoml.validation.base import TrainValidIterator
from pyspark.sql import functions as sf

from sparklightautoml import VALIDATION_COLUMN
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.utils import SparkDataFrame

TrainVal = Tuple[SparkDataset, SparkDataset]


class SparkSelectionPipeline(SelectionPipeline, ABC):
    def __init__(self,
                 features_pipeline: Optional[FeaturesPipeline] = None,
                 ml_algo: Optional[Union[MLAlgo, Tuple[MLAlgo, ParamsTuner]]] = None,
                 imp_estimator: Optional[ImportanceEstimator] = None,
                 fit_on_holdout: bool = False,
                 **kwargs: Any):
        super().__init__(features_pipeline, ml_algo, imp_estimator, fit_on_holdout, **kwargs)


class SparkBaseTrainValidIterator(TrainValidIterator, ABC):
    """
    Implements applying selection pipeline and feature pipeline to SparkDataset.
    """

    TRAIN_VAL_COLUMN = VALIDATION_COLUMN

    def __init__(self, train: SparkDataset):
        assert train.folds_column in train.data.columns
        super().__init__(train)
        self.train = cast(SparkDataset, train)

    def __next__(self) -> TrainVal:
        """Define how to get next object.

        Returns:
            a tuple with:
            - train part of the dataset
            - validation part of the dataset.

        """
        ...

    @abstractmethod
    @property
    def train_frozen(self) -> bool:
        ...

    @abstractmethod
    @train_frozen.setter
    def train_frozen(self, val: bool):
        ...

    @abstractmethod
    @property
    def val_frozen(self) -> bool:
        ...

    @abstractmethod
    @val_frozen.setter
    def val_frozen(self, val: bool):
        ...

    @abstractmethod
    def unpersist(self):
        ...

    @contextmanager
    def _child_persistence_context(self) -> 'SparkBaseTrainValidIterator':
        train_valid = copy(self)
        train = train_valid.train.empty()
        child_manager = train_valid.train.persistence_manager.child()
        train.set_data(
            train_valid.train.data,
            train_valid.train.features,
            train_valid.train.roles,
            persistence_manager=child_manager,
            dependencies=None
        )
        train_valid.train = train

        yield train_valid

        child_manager.unpersist_all()

    def apply_selector(self, selector: SparkSelectionPipeline) -> "SparkBaseTrainValidIterator":
        """Select features on train data.

        Check if selector is fitted.
        If not - fit and then perform selection.
        If fitted, check if it's ok to apply.

        Args:
            selector: Uses for feature selection.

        Returns:
            Dataset with selected features.

        """
        if not selector.is_fitted:
            with self._child_persistence_context() as sel_train_valid:
                selector.fit(sel_train_valid)

        train_valid = copy(self)
        train_valid.train = selector.select(cast(SparkDataset, self.train))

        return train_valid

    def apply_feature_pipeline(
            self,
            features_pipeline: SparkFeaturesPipeline) -> "SparkBaseTrainValidIterator":
        train_valid = copy(self)
        train_valid.train = features_pipeline.fit_transform(train_valid.train)

        return train_valid

    def combine_val_preds(self, val_preds: Sequence[SparkDataFrame]) -> SparkDataFrame:
        # depending on train_valid logic there may be several ways of treating predictions results:
        # 1. for folds iterators - join the results, it will yield the full train dataset
        # 2. for holdout iterators - create None predictions in train_part and join with valid part
        # 3. for custom iterators which may put the same records in
        #   different folds: union + groupby + (optionally) union with None-fied train_part
        # 4. for dummy - do nothing
        raise NotImplementedError()

    def _split_by_fold(self, fold: int) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        train = cast(SparkDataset, self.train)
        is_val_col = (
            sf.when(sf.col(self.train.folds_column) != fold, sf.lit(0)).otherwise(sf.lit(1))
            .alias(self.TRAIN_VAL_COLUMN)
        )

        sdf = train.data.select("*", is_val_col)
        train_part_sdf = sdf.where(sf.col(self.TRAIN_VAL_COLUMN) == 0).drop(self.TRAIN_VAL_COLUMN)
        valid_part_sdf = sdf.where(sf.col(self.TRAIN_VAL_COLUMN) == 1).drop(self.TRAIN_VAL_COLUMN)

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(sdf, self.train.features, self.train.roles, name=self.train.name)

        train_part_ds = cast(SparkDataset, self.train.empty())
        train_part_ds.set_data(
            train_part_sdf,
            self.train.features,
            self.train.roles,
            name=f"{self.train.name}_train_{fold}"
        )

        valid_part_ds = cast(SparkDataset, self.train.empty())
        valid_part_ds.set_data(
            valid_part_sdf,
            self.train.features,
            self.train.roles,
            name=f"{self.train.name}_val_{fold}"
        )

        return train_ds, train_part_ds, valid_part_ds
