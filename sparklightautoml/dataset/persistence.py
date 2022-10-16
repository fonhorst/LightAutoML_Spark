import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, cast, List

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.utils import SparkDataFrame

PersistenceIdentifable = Union[str, SparkDataset]


@dataclass(frozen=True)
class PersistableDataset:
    dataset: SparkDataset
    callback: Optional[Callable] = None

    @property
    def uid(self) -> str:
        return self.dataset.uid


@dataclass(frozen=True)
class PersistableDataFrame:
    sdf: SparkDataFrame
    uid: str
    callback: Optional[Callable] = None
    base_dataset: Optional[SparkDataset] = None

    def to_dataset(self) -> SparkDataset:
        assert self.base_dataset
        ds = self.base_dataset.empty()
        ds.set_data(
            self.sdf,
            self.base_dataset.features,
            self.base_dataset.roles,
            dependencies=list(self.base_dataset.dependencies),
            uid=self.uid
        )
        return ds


# TODO: SLAMA - add documentation
class PersistenceManager(ABC):
    @staticmethod
    def to_persistable_dataframe(dataset: SparkDataset) -> PersistableDataFrame:
        # we intentially create new uid to use to distinguish a persisted and unpersisted dataset
        return PersistableDataFrame(dataset.data, uid=str(uuid.uuid4()), dataset=dataset)

    def __init__(self, parent: Optional['PersistenceManager'] = None):
        self._persistence_registry: Dict[str, PersistableDataFrame] = dict()
        self._parent = parent
        self._children: List['PersistenceManager'] = []

    def persist(self, dataset: Union[SparkDataset, PersistableDataFrame]) -> PersistableDataFrame:
        persisted_dataframe = self.to_persistable_dataframe(dataset) if isinstance(dataset, SparkDataset) \
            else cast(PersistableDataFrame, dataset)

        if persisted_dataframe.uid in self._persistence_registry:
            return self._persistence_registry[persisted_dataframe.uid]

        persisted_dataframe = self._persist(persisted_dataframe)
        self._persistence_registry[persisted_dataframe.uid] = persisted_dataframe

        return persisted_dataframe

    def unpersist(self, uid: str):
        persisted_dataframe = self._persistence_registry.get(uid, None)

        if not persisted_dataframe:
            return

        self._unpersist(persisted_dataframe)

        del self._persistence_registry[persisted_dataframe.uid]

    def unpersist_all(self):
        self.unpersist_children()

        uids = list(self._persistence_registry.keys())

        for uid in uids:
            self.unpersist(uid)

    def unpersist_children(self):
        for child in self._children:
            child.unpersist_all()
        self._children = []

    def child(self) -> 'PersistenceManager':
        a_child = PersistenceManager(self)
        self._children.append(a_child)
        return a_child

    @abstractmethod
    def _persist(self, pdf: PersistableDataFrame) -> PersistableDataFrame:
        ...

    @abstractmethod
    def _unpersist(self, pdf: PersistableDataFrame):
        ...


class PlainCachePersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame) -> PersistableDataFrame:
        ds = SparkSession.getActiveSession().createDataFrame(pdf.sdf.rdd, schema=pdf.sdf.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class LocalCheckpointPersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame) -> PersistableDataFrame:
        ds = pdf.sdf.localCheckpoint()
        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class BucketedPersistenceManager(PersistenceManager):
    def __init__(self, bucketed_datasets_folder: str, bucket_nums: int = 100, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)
        self._bucket_nums = bucket_nums
        self._bucketed_datasets_folder = bucketed_datasets_folder

    def _persist(self, pdf: PersistableDataFrame) -> PersistableDataFrame:
        spark = SparkSession.getActiveSession()
        name = "SparkToSparkReaderTable"
        # TODO: SLAMA join - need to identify correct setting  for bucket_nums if it is not provided
        (
            pdf.sdf
            .repartition(self._bucket_nums, SparkDataset.ID_COLUMN)
            .write
            .mode('overwrite')
            .bucketBy(self._bucket_nums, SparkDataset.ID_COLUMN)
            .sortBy(SparkDataset.ID_COLUMN)
            .saveAsTable(name, format='parquet', path=self._build_path(pdf.uid))
        )
        ds = spark.table(name)

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        path = self._build_path(pdf.uid)
        # TODO: SLAMA - add local file removing
        # TODO: SLAMA - add file removing on hdfs
        raise NotImplementedError()

    def _build_path(self, uid: str) -> str:
        return os.path.join(self._bucketed_datasets_folder, f"{uid}.parquet")
