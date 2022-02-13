from abc import ABC
from typing import Optional, Iterable, Tuple, cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.spark import VALIDATION_COLUMN
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.validation.base import TrainValidIterator

from pyspark.sql import functions as F


class SparkBaseTrainValidIterator(TrainValidIterator, ABC):
    TRAIN_VAL_COLUMN = VALIDATION_COLUMN

    def __init__(self, train: SparkDataset):
        assert train.folds_column in train.data.columns
        super().__init__(train)

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