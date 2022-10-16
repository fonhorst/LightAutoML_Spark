from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, cast, List

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset

PersistenceIdentifable = Union[str, SparkDataset]


@dataclass(frozen=True)
class PersistedDataset:
    dataset: SparkDataset
    custom_persistence: bool = False
    callback: Optional[Callable] = None


# TODO: SLAMA - add documentation
class PersistenceManager:
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        self._persistence_registry: Dict[str, PersistedDataset] = dict()
        self._persisted_datasets: Dict[str, str] = dict()
        self._parent = parent
        self._children: List['PersistenceManager'] = []

    def persist(self, dataset: Union[SparkDataset, PersistedDataset], *, name: Optional[str] = None) -> SparkDataset:
        # assert dataset.uid not in self._persisted_datasets or self._persisted_datasets[dataset.uid] == name, \
        #     f"Cannot persist the same dataset with different names. Called with name: {name}. " \
        #     f"Already exists in the registry: {self._persisted_datasets[dataset.uid]}"

        if isinstance(dataset, SparkDataset):
            name = dataset.uid if name is None else name
            ds = SparkSession.getActiveSession().createDataFrame(dataset.data.rdd, schema=dataset.data.schema).cache()
            ds.write.mode('overwrite').format('noop').save()

            # TODO: need to save info about how it has been cached
            ps_dataset = dataset.empty()
            ps_dataset.set_data(ds, dataset.features, dataset.roles)
            persisted_dataset = PersistedDataset(ps_dataset)
        else:
            persisted_dataset = cast(PersistedDataset, dataset)
            name = persisted_dataset.dataset.uid if name is None else name

        self.unpersist(name)

        self._persistence_registry[name] = persisted_dataset
        self._persisted_datasets[persisted_dataset.dataset.uid] = name

        return persisted_dataset.dataset

    def unpersist(self, name_or_dataset: PersistenceIdentifable):
        name = name_or_dataset if isinstance(name_or_dataset, str) \
            else self._persisted_datasets.get(name_or_dataset.uid, None)

        persisted_dataset = self._persistence_registry.get(name, None)

        if not persisted_dataset:
            return

        if not persisted_dataset.custom_persistence:
            persisted_dataset.dataset.data.unpersist()

        persisted_dataset.callback()

        del self._persisted_datasets[persisted_dataset.dataset.uid]
        del self._persistence_registry[name]

    def unpersist_all(self, exceptions: Optional[Union[PersistenceIdentifable, List[PersistenceIdentifable]]] = None):
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
