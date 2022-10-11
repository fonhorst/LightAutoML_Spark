from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, cast

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset


class CacheAware(ABC):
    def release_cache(self):
        ...


@dataclass(frozen=True)
class PersistedDataset:
    dataset: SparkDataset
    custom_persistence: bool = False
    callback: Optional[Callable] = None


class PersistenceManager:
    def __init__(self):
        self._persistence_registry: Dict[str, PersistedDataset] = dict()

    def persist(self, dataset: Union[SparkDataset, PersistedDataset], *, name: str) -> SparkDataset:
        if isinstance(dataset, SparkDataset):
            ds = SparkSession.getActiveSession().createDataFrame(dataset.data.rdd, schema=dataset.data.schema).cache()
            ds.write.mode('overwrite').format('noop').save()

            # TODO: need to save info about how it has been cached
            ps_dataset = dataset.empty()
            ps_dataset.set_data(ds, dataset.features, dataset.roles)
            persisted_dataset = PersistedDataset(ps_dataset)
        else:
            persisted_dataset = cast(PersistedDataset, dataset)

        self.unpersist(name)

        self._persistence_registry[name] = persisted_dataset

        return persisted_dataset.dataset

    def unpersist(self, name: str):
        if name not in self._persistence_registry:
            return

        persisted_dataset = self._persistence_registry[name]
        if not persisted_dataset.custom_persistence:
            persisted_dataset.dataset.data.unpersist()

        persisted_dataset.callback()

        del self._persistence_registry[name]

    def unpersist_all(self):
        for name in self._persistence_registry:
            self.unpersist(name)
