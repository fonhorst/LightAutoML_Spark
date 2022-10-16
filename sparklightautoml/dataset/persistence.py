import uuid
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
        ds.set_data(self.sdf, self.base_dataset.features, self.base_dataset.roles, self.base_dataset.dependencies)
        return ds


# TODO: SLAMA - add documentation
class PersistenceManager:

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

        if not persisted_dataframe.callback:
            deps = persisted_dataframe.base_dataset.dependencies
            persisted_dataframe = self._persist(persisted_dataframe)
            for dep in deps:
                self.unpersist(dep)

        self._persistence_registry[persisted_dataframe.uid] = persisted_dataframe

        return persisted_dataframe

    def unpersist(self, uid: str):
        persisted_dataset = self._persistence_registry.get(uid, None)

        if not persisted_dataset:
            return

        if persisted_dataset.callback:
            persisted_dataset.callback()
        else:
            self._unpersist(persisted_dataset)

        del self._persistence_registry[persisted_dataset.uid]

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
    
    def _persist(self, pdf: PersistableDataFrame) -> PersistableDataFrame:
        ds = SparkSession.getActiveSession().createDataFrame(pdf.sdf.rdd, schema=pdf.sdf.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()
