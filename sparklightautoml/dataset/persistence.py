from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, cast, List

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, Dependency, DepIdentifable
from sparklightautoml.utils import SparkDataFrame

PersistenceIdentifable = Union[str, SparkDataset]


@dataclass(frozen=True)
class PersistableDataset:
    dataset: SparkDataset
    custom_persistence: bool = False
    callback: Optional[Callable] = None

    @property
    def uid(self) -> str:
        return self.dataset.uid


# TODO: SLAMA - add documentation
class PersistenceManager:
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        self._persistence_registry: Dict[str, PersistableDataset] = dict()
        self._persisted_datasets: Dict[str, str] = dict()
        self._parent = parent
        self._children: List['PersistenceManager'] = []

    def persist(self, dataset: Union[SparkDataset, PersistableDataset]) -> SparkDataset:
        # assert dataset.uid not in self._persisted_datasets or self._persisted_datasets[dataset.uid] == name, \
        #     f"Cannot persist the same dataset with different names. Called with name: {name}. " \
        #     f"Already exists in the registry: {self._persisted_datasets[dataset.uid]}"

        persisted_dataset = PersistableDataset(dataset) if isinstance(dataset, SparkDataset) \
            else cast(PersistableDataset, dataset)

        if persisted_dataset.uid in self._persistence_registry:
            return self._persistence_registry[persisted_dataset.uid].dataset

        if not persisted_dataset.custom_persistence:
            deps = persisted_dataset.dataset.dependencies
            persisted_dataset = self._persist(persisted_dataset.dataset)
            for dep in deps:
                self.unpersist(dep)

        self._persistence_registry[persisted_dataset.uid] = persisted_dataset

        return persisted_dataset.dataset

    def unpersist(self, identifier: DepIdentifable):
        uid = identifier.uid if isinstance(identifier, SparkDataset) else cast(str, identifier)
        persisted_dataset = self._persistence_registry.get(uid, None)

        if not persisted_dataset:
            return

        if persisted_dataset.custom_persistence:
            persisted_dataset.callback()
        else:
            self._unpersist(persisted_dataset)

        del self._persistence_registry[persisted_dataset.uid]

    def unpersist_all(self, exceptions: Optional[Union[DepIdentifable, List[DepIdentifable]]] = None):
        if exceptions:
            if not isinstance(exceptions, list):
                exceptions = [exceptions]

            names_to_save = {
                ex if isinstance(ex, str) else self._persisted_datasets.get(ex.uid, None)
                for ex in exceptions
            }
            names = [name for name in self._persistence_registry if name not in names_to_save]
        else:
            names = list(self._persistence_registry)

        self.unpersist_children()

        for name in names:
            self.unpersist(name)

    def unpersist_children(self):
        for child in self._children:
            child.unpersist_all()
        self._children = []

    def child(self) -> 'PersistenceManager':
        a_child = PersistenceManager(self)
        self._children.append(a_child)
        return a_child
    
    def _persist(self, dataset: SparkDataset) -> PersistableDataset:
        ds = SparkSession.getActiveSession().createDataFrame(dataset.data.rdd, schema=dataset.data.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        # TODO: need to save info about how it has been cached
        ps_dataset = dataset.empty()
        ps_dataset.set_data(ds, dataset.features, dataset.roles)

        return PersistableDataset(dataset)

    def _unpersist(self, persisted_dataset: PersistableDataset):
        persisted_dataset.dataset.data.unpersist()
