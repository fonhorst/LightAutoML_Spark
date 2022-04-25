import functools
import logging
from typing import Optional, cast, Tuple, Iterable, Sequence

from lightautoml.dataset.base import LAMLDataset, RolesDict
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.transformers.scala_wrappers.balanced_union_partitions_coalescer import \
    BalancedUnionPartitionsCoalescerTransformer
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator
from lightautoml.validation.base import TrainValidIterator, HoldoutIterator

from pyspark.sql import functions as F


logger = logging.getLogger(__name__)


class SparkDummyIterator(SparkBaseTrainValidIterator):
    def __init__(self, train: SparkDataset, input_roles: Optional[RolesDict] = None):
        super().__init__(train, input_roles)
        self._curr_idx = 0

    def __iter__(self) -> Iterable:
        self._curr_idx = 0
        return self

    def __len__(self) -> Optional[int]:
        return 1

    def __next__(self) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        if self._curr_idx > 0:
            raise StopIteration

        self._curr_idx += 1

        sdf = cast(SparkDataFrame, self.train.data)
        sdf = sdf.withColumn(self.TRAIN_VAL_COLUMN, F.lit(0))

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(sdf, self.train.features, self.train.roles)

        return train_ds, train_ds, train_ds

    def combine_val_preds(self, val_preds: Sequence[SparkDataFrame], include_train: bool = False) -> SparkDataFrame:
        assert len(val_preds) == 1
        return val_preds[0]

    def get_validation_data(self) -> SparkDataset:
        return self.train

    def convert_to_holdout_iterator(self) -> "SparkHoldoutIterator":
        sds = cast(SparkDataset, self.train)
        assert sds.folds_column is not None, \
            "Cannot convert to Holdout iterator when folds_column is not defined"
        return SparkHoldoutIterator(self.train, self.input_roles)


class SparkHoldoutIterator(SparkBaseTrainValidIterator):
    def __init__(self, train: SparkDataset, input_roles: Optional[RolesDict] = None):
        super().__init__(train, input_roles)
        self._curr_idx = 0

    def __iter__(self) -> Iterable:
        self._curr_idx = 0
        return self

    def __len__(self) -> Optional[int]:
        return 1

    def __next__(self) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        if self._curr_idx > 0:
            raise StopIteration

        full_ds, train_part_ds, valid_part_ds = self._split_by_fold(self._curr_idx)
        self._curr_idx += 1

        return full_ds, train_part_ds, valid_part_ds

    def get_validation_data(self) -> SparkDataset:
        full_ds, train_part_ds, valid_part_ds = self._split_by_fold(fold=0)
        return valid_part_ds

    def convert_to_holdout_iterator(self) -> "SparkHoldoutIterator":
        return self

    def combine_val_preds(self, val_preds: Sequence[SparkDataFrame], include_train: bool = False) -> SparkDataFrame:
        if len(val_preds) != 1:
            k = 0
        assert len(val_preds) == 1

        if not include_train:
            return val_preds[0]

        val_pred_cols = set(val_preds[0].columns)
        train_cols = set(self.train.columns)
        assert len(train_cols.difference(val_pred_cols)) == 0
        new_feats = val_pred_cols.difference(train_cols)

        _, train_ds, _ = self._split_by_fold(0)
        missing_cols = [F.lit(None).alias(f) for f in new_feats]
        full_val_preds = train_ds.select('*', *missing_cols).unionByName(val_preds[0])

        return full_val_preds


class SparkFoldsIterator(SparkBaseTrainValidIterator):
    """Classic cv iterator.

    Folds should be defined in Reader, based on cross validation method.
    """

    def __init__(self, train: SparkDataset, n_folds: Optional[int] = None, input_roles: Optional[RolesDict] = None, seed: Optional[float] = None):
        """Creates iterator.

        Args:
            train: Dataset for folding.
            n_folds: Number of folds.

        """
        # TODO: SPARK-LAMA for debug only, remove later
        if seed is not None:
            t_df = (
                train.data
                .drop(train.folds_column)
                .select('*', F.floor(F.rand(seed) * n_folds).astype('int').alias(train.folds_column))
            )
            new_train = train.empty()
            new_train.set_data(t_df, train.features, train.roles)
            train = new_train

        super().__init__(train, input_roles)

        num_folds = train.data.select(F.max(train.folds_column).alias('max')).first()['max']
        self.n_folds = num_folds + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

    def __len__(self) -> int:
        """Get len of iterator.

        Returns:
            Number of folds.

        """
        return self.n_folds

    def __iter__(self) -> "SparkFoldsIterator":
        """Set counter to 0 and return self.

        Returns:
            Iterator for folds.

        """
        logger.debug("Creating folds iterator")

        self._curr_idx = 0

        return self

    def __next__(self) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        logger.debug(f"The next valid fold num: {self._curr_idx}")

        if self._curr_idx == self.n_folds:
            logger.debug("No more folds to continue, stopping iterations")
            raise StopIteration

        full_ds, train_part_ds, valid_part_ds = self._split_by_fold(self._curr_idx)
        self._curr_idx += 1

        return full_ds, train_part_ds, valid_part_ds

    def get_validation_data(self) -> SparkDataset:
        """Just return train dataset.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> SparkHoldoutIterator:
        """Convert iterator to hold-out-iterator.

        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new hold-out-iterator.

        """
        return SparkHoldoutIterator(self.train, self.input_roles)

    def combine_val_preds(self, val_preds: Sequence[SparkDataFrame], include_train: bool = False) -> SparkDataFrame:
        assert len(val_preds) > 0

        if len(val_preds) == 1:
            return val_preds[0]

        num_partitions = val_preds[0].rdd.getNumPartitions()
        full_val_preds = functools.reduce(lambda x, y: x.unionByName(y), val_preds)
        # TODO: SPARK-LAMA for debug only
        import pprint
        def fun2(index, p):
            from collections import Counter
            acc = 0
            folds = Counter()
            for row in p:
                acc += 1
                folds.update([row['reader_fold_num']])
            yield index, acc, list(folds.items())

        res = full_val_preds.rdd.mapPartitionsWithIndex(fun2).collect()
        print(f"#1 PARTITIONS AFTER UNION")
        pprint.pprint(res)

        res = full_val_preds.coalesce(num_partitions).rdd.mapPartitionsWithIndex(fun2).collect()
        print(f"#2 PARTITIONS AFTER UNION and COALESCE")
        pprint.pprint(res)

        # # TODO: SPARK-LAMA temporary method of preventing of uneven data distribution
        # # hack to prevent uneven distribution of partitions
        # # and data among partitions after coalesce
        # full_val_preds = full_val_preds.localCheckpoint(True)
        # full_val_preds = full_val_preds.coalesce(num_partitions)
        #
        # res = full_val_preds.rdd.mapPartitionsWithIndex(fun2).collect()
        # print(f"#3 PARTITIONS AFTER UNION and LOCALCHEKPOINT with COALESCE")
        # pprint.pprint(res)

        full_val_preds = BalancedUnionPartitionsCoalescerTransformer().transform(full_val_preds)

        res = full_val_preds.rdd.mapPartitionsWithIndex(fun2).collect()
        print(f"#4 PARTITIONS AFTER UNION and CUSTOM COALESCER")
        pprint.pprint(res)

        return full_val_preds
