from abc import ABC
from copy import copy
from typing import Tuple, cast, Optional, List

from pyspark.sql import functions as F

from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.reader.base import RolesDict
from lightautoml.spark import VALIDATION_COLUMN
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.base import InputFeaturesAndRoles
from lightautoml.spark.pipelines.features.base import SparkFeaturesPipeline
from lightautoml.validation.base import TrainValidIterator


class SparkBaseTrainValidIterator(TrainValidIterator, InputFeaturesAndRoles, ABC):
    TRAIN_VAL_COLUMN = VALIDATION_COLUMN

    def __init__(self, train: SparkDataset, input_roles: Optional[RolesDict] = None):
        assert train.folds_column in train.data.columns
        super().__init__(train)
        if not input_roles:
            input_roles = train.roles
        self._input_roles = input_roles

    @property
    def features(self) -> List[str]:
        return self.input_features

    def apply_selector(self, selector: SelectionPipeline) -> "TrainValidIterator":
        """Select features on train data.

        Check if selector is fitted.
        If not - fit and then perform selection.
        If fitted, check if it's ok to apply.

        Args:
            selector: Uses for feature selection.

        Returns:
            Dataset with selected features.

        """
        # TODO: SPARK-LAMA selector should have cacher with different name
        # TODO: SPARK-LAMA cacher should be cleared here
        if not selector.is_fitted:
            selector.fit(self)
        train_valid = copy(self)
        train_valid.input_roles = {feat: self.input_roles[feat]
                                   for feat in selector.selected_features}
        return train_valid

    def apply_feature_pipeline(self, features_pipeline: SparkFeaturesPipeline) -> "TrainValidIterator":
        features_pipeline.input_roles = self.input_roles
        train_valid = cast(SparkBaseTrainValidIterator, super().apply_feature_pipeline(features_pipeline))
        train_valid.input_roles = features_pipeline.output_roles
        return train_valid

    def _split_by_fold(self, fold: int) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        train = cast(SparkDataset, self.train)
        is_val_col = (
            F.when(F.col(self.train.folds_column) != fold, F.lit(0))
            .otherwise(F.lit(1))
            .alias(self.TRAIN_VAL_COLUMN)
        )

        sdf = train.data.select('*', is_val_col)
        train_part_sdf = sdf.where(F.col(self.TRAIN_VAL_COLUMN) == 0)
        valid_part_sdf = sdf.where(F.col(self.TRAIN_VAL_COLUMN) == 1)

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(sdf, self.train.features, self.train.roles)

        train_part_ds = cast(SparkDataset, self.train.empty())
        train_part_ds.set_data(train_part_sdf, self.train.features, self.train.roles)

        valid_part_ds = cast(SparkDataset, self.train.empty())
        valid_part_ds.set_data(valid_part_sdf, self.train.features, self.train.roles)

        return train_ds, train_part_ds, valid_part_ds