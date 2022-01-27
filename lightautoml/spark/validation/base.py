"""Basic classes for validation iterators."""

from copy import copy
from typing import Any, Generator, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.pipelines.features.base import FeaturesPipeline


# from ..pipelines.selection.base import SelectionPipeline

# TODO: SOLVE CYCLIC IMPORT PROBLEM!!! add Selectors typing

# Dataset = TypeVar("Dataset", bound=LAMLDataset)
CustomIdxs = Iterable[Tuple[Sequence, Sequence]]



class CustomIterator(TrainValidIterator):
    """Iterator that uses function to create folds indexes.

    Usefull for example - classic timeseries splits.

    """

    def __init__(self, train: SparkDataset, iterator: CustomIdxs):
        """Create iterator.

        Args:
            train: Dataset of train data.
            iterator: Callable(dataset) -> Iterator of train/valid indexes.

        """
        self.train = train
        self.iterator = iterator

    def __len__(self) -> Optional[int]:
        """Empty __len__ method.

        Returns:
            None.

        """

        return len(self.iterator)

    def __iter__(self) -> Generator:
        """Create generator of train/valid datasets.

        Returns:
            Data generator.

        """
        generator = ((val_idx, self.train[tr_idx], self.train[val_idx]) for (tr_idx, val_idx) in self.iterator)

        return generator

    def get_validation_data(self) -> SparkDataset:
        """Simple return train dataset.

        Returns:
            Dataset of train data.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Convert iterator to hold-out-iterator.

        Use first train/valid split for :class:`~lightautoml.validation.base.HoldoutIterator` creation.

        Returns:
            New hold out iterator.

        """
        for (tr_idx, val_idx) in self.iterator:
            return HoldoutIterator(self.train[tr_idx], self.train[val_idx])
