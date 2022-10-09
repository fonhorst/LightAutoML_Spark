from abc import ABC
from typing import Dict

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset


class CacheAware(ABC):
    def release_cache(self):
        ...


class CacheManager:
    def __init__(self):
        self._milestone_registry: Dict[str, SparkDataset] = dict()

    def milestone(self, dataset: SparkDataset, *, name: str) -> SparkDataset:
        ds = SparkSession.getActiveSession().createDataFrame(dataset.data.rdd, schema=dataset.data.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        # TODO: need to save info about how it has been cached
        milestone_dataset = dataset.empty()
        milestone_dataset.set_data(ds, dataset.features, dataset.roles)

        if name in self._milestone_registry:
            self.remove_milestone(name)

        self._milestone_registry[name] = milestone_dataset

        return milestone_dataset

    def remove_milestone(self, name: str):
        assert name in self._milestone_registry

        dataset = self._milestone_registry[name]
        dataset.data.unpersist()

        del self._milestone_registry[name]

    def remove_all(self):
        for name in self._milestone_registry:
            self.remove_milestone(name)
