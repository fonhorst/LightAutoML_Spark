import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, cast, List

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, Dependency
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
            dependencies=self.base_dataset.dependencies,
            uid=self.uid
        )
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

        persisted_dataframe = self._persist(persisted_dataframe)

        if persisted_dataframe.callback:
            persisted_dataframe.callback()
        elif persisted_dataframe.base_dataset and persisted_dataframe.base_dataset.dependencies:
            deps = persisted_dataframe.base_dataset.dependencies
            for dep in deps:
                self.unpersist(dep)

        self._persistence_registry[persisted_dataframe.uid] = persisted_dataframe

        return persisted_dataframe

    def unpersist(self, dep: Dependency):
        if isinstance(dep, str):
            uid = cast(str, dep)
        elif isinstance(dep, SparkDataset) and dep.frozen:
            return
        elif isinstance(dep, SparkDataset):
            uid = dep.uid
        elif isinstance(dep, PersistableDataFrame):
            uid = dep.uid
        else:
            dep()
            return

        persisted_dataframe = self._persistence_registry.get(uid, None)

        if not persisted_dataframe:
            return

        self._unpersist(persisted_dataframe)

        del self._persistence_registry[persisted_dataframe.uid]

        if isinstance(dep, SparkDataset):
            for dep in (dep.dependencies or []):
                self.unpersist(dep)

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
