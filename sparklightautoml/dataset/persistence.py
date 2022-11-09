import logging
import os
import shutil
import uuid
import warnings
from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, cast

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, PersistenceLevel, PersistableDataFrame, PersistenceManager
from sparklightautoml.utils import JobGroup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PersistablePair:
    level: PersistenceLevel
    pdf: PersistableDataFrame


class BasePersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        self._uid = str(uuid.uuid4())
        self._persistence_registry: Dict[str, _PersistablePair] = dict()
        self._parent = parent
        self._children: Dict[str, 'PersistenceManager'] = dict()

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def children(self) -> List['PersistenceManager']:
        return list(self._children.values())

    @property
    def datasets(self) -> List[SparkDataset]:
        return [pair.pdf.to_dataset() for pair in self._persistence_registry.values()]

    @property
    def all_datasets(self) -> List[SparkDataset]:
        return [*self.datasets, *(ds for child in self.children for ds in child.all_datasets)]

    def persist(self,
                dataset: Union[SparkDataset, PersistableDataFrame],
                level: PersistenceLevel = PersistenceLevel.REGULAR) -> PersistableDataFrame:
        persisted_dataframe = self.to_persistable_dataframe(dataset) if isinstance(dataset, SparkDataset) \
            else cast(PersistableDataFrame, dataset)

        logger.debug(f"Manager {self._uid}: "
                     f"persisting dataset (uid={dataset.uid}, name={dataset.name}) with level {level}.")

        if persisted_dataframe.uid in self._persistence_registry:
            persisted_pair = self._persistence_registry[persisted_dataframe.uid]

            logger.debug(f"Manager {self._uid}: "
                         f"the dataset (uid={dataset.uid}, name={dataset.name}) is already persisted.")

            if persisted_pair.level != level:
                warnings.warn(
                    f"Asking to persist an already persisted dataset "
                    f"(uid={persisted_pair.pdf.uid}, name={persisted_pair.pdf.name}) "
                    f"but with different level. Will do nothing. "
                    f"Current level {persisted_pair.level}, asked level {level}.",
                    RuntimeWarning
                )

            return persisted_pair.pdf

        self._persistence_registry[persisted_dataframe.uid] = \
            _PersistablePair(level, self._persist(persisted_dataframe, level))

        logger.debug(f"Manager {self._uid}: the dataset (uid={dataset.uid}, name={dataset.name}) has been persisted.")

        return self._persistence_registry[persisted_dataframe.uid].pdf

    def unpersist(self, uid: str):
        persisted_pair = self._persistence_registry.get(uid, None)

        if not persisted_pair:
            logger.debug(f"Manager {self._uid} unpersist: the dataset (uid={uid}) is not persisted yet. Nothing to do.")
            return

        logger.debug(f"Manager {self._uid}: unpersisting dataset (uid={uid}, name={persisted_pair.pdf.name}).")

        self._unpersist(persisted_pair.pdf)

        del self._persistence_registry[persisted_pair.pdf.uid]

        logger.debug(f"Manager {self._uid}: "
                     f"the dataset (uid={uid}, name={persisted_pair.pdf.name}) has been unpersisted.")

    def unpersist_children(self):
        logger.info(f"Manager {self._uid}: unpersisting children.")

        for child in self._children:
            child.unpersist_all()
        self._children = dict()

        logger.debug(f"Manager {self._uid}: children have been unpersisted.")

    def unpersist_all(self):
        logger.info(f"Manager {self._uid}: unpersisting everything.")

        self.unpersist_children()

        uids = list(self._persistence_registry.keys())

        for uid in uids:
            self.unpersist(uid)

        logger.debug(f"Manager {self._uid}: everything has been unpersisted.")

    def child(self) -> 'PersistenceManager':
        # TODO: SLAMA - Bugfix
        logger.info(f"Manager {self._uid}: producing a child.")
        a_child = self._create_child()
        self._children[a_child.uid] = a_child
        logger.info(f"Manager {self._uid}: the child (uid={a_child.uid}) has been produced.")
        return a_child

    def remove_child(self, child: Union['PersistenceManager', str]):
        uid = child.uid if isinstance(child, PersistenceManager) else child

        if uid not in self._children:
            logger.warning(f"Not found child with uid {uid} to delete from parent {self._uid}")
            return

        del self._children[uid]

    def is_persisted(self, pdf: PersistableDataFrame) -> bool:
        return pdf.uid in self._persistence_registry

    @abstractmethod
    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        ...

    @abstractmethod
    def _unpersist(self, pdf: PersistableDataFrame):
        ...

    @abstractmethod
    def _create_child(self) -> PersistenceManager:
        ...


class PlainCachePersistenceManager(BasePersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None, prune_history: bool = False):
        super().__init__(parent)
        self._prune_history = prune_history

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        logger.debug(f"Manager {self._uid}: "
                     f"caching and materializing the dataset (uid={pdf.uid}, name={pdf.name}).")

        with JobGroup("Persisting", f"{type(self)} caching df (uid={pdf.uid}, name={pdf.name})"):
            df = SparkSession.getActiveSession().createDataFrame(pdf.sdf.rdd, schema=pdf.sdf.schema) \
                if self._prune_history else pdf.sdf
            ds = df.cache()
            ds.write.mode('overwrite').format('noop').save()

        logger.debug(f"Manager {self._uid}: "
                     f"caching succeeded for the dataset (uid={pdf.uid}, name={pdf.name}).")

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()

    def _create_child(self) -> PersistenceManager:
        return PlainCachePersistenceManager(self)


class LocalCheckpointPersistenceManager(BasePersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        logger.debug(f"Manager {self._uid}: "
                     f"making a local checkpoint for the dataset (uid={pdf.uid}, name={pdf.name}).")

        with JobGroup("Persisting", f"{type(self)} local checkpointing of df (uid={pdf.uid}, name={pdf.name})"):
            ds = pdf.sdf.localCheckpoint()

        logger.debug(f"Manager {self._uid}: "
                     f"the local checkpoint has been made for the dataset (uid={pdf.uid}, name={pdf.name}).")
        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()

    def _create_child(self) -> PersistenceManager:
        return LocalCheckpointPersistenceManager(self)


class BucketedPersistenceManager(BasePersistenceManager):
    def __init__(self,
                 bucketed_datasets_folder: str,
                 bucket_nums: int = 100,
                 parent: Optional['PersistenceManager'] = None,
                 no_unpersisting: bool = False):
        super().__init__(parent)
        self._bucket_nums = bucket_nums
        self._bucketed_datasets_folder = bucketed_datasets_folder
        self._no_unpersisting = no_unpersisting

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        spark = SparkSession.getActiveSession()
        name = self._build_name(pdf)
        # TODO: SLAMA join - need to identify correct setting  for bucket_nums if it is not provided
        path = self._build_path(name)
        logger.debug(
            f"Manager {self._uid}: making a bucketed table "
            f"for the dataset (uid={pdf.uid}, name={pdf.name}) with name {name} on path {path}."
        )

        with JobGroup("Persisting",
                      f"{type(self)} saving bucketed table of df (uid={pdf.uid}, name={pdf.name}). Table path: {path}"):
            (
                pdf.sdf
                .repartition(self._bucket_nums, SparkDataset.ID_COLUMN)
                .write
                .mode('overwrite')
                .bucketBy(self._bucket_nums, SparkDataset.ID_COLUMN)
                .sortBy(SparkDataset.ID_COLUMN)
                .saveAsTable(name, format='parquet', path=path)
            )
        ds = spark.table(name)

        logger.debug(
            f"Manager {self._uid}: the bucketed table has been made "
            f"for the dataset (uid={pdf.uid}, name={pdf.name}) with name {name} on path {path}."
        )
        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        if self._no_unpersisting:
            return

        name = self._build_name(pdf)
        path = self._build_path(name)
        logger.debug(
            f"Manager {self._uid}: removing the bucketed table "
            f"for the dataset (uid={pdf.uid}, name={pdf.name}) with name {name} on path {path}."
        )

        # TODO: SLAMA - add file removing on hdfs
        # TODO: SLAMA - only local path is supported
        shutil.rmtree(path)

        logger.debug(
            f"Manager {self._uid}: the bucketed table has been removed"
            f"for the dataset (uid={pdf.uid}, name={pdf.name}) with name {name} on path {path}."
        )

    def _create_child(self) -> PersistenceManager:
        return BucketedPersistenceManager(self._bucketed_datasets_folder, self._bucket_nums, self)

    def _build_path(self, name: str) -> str:
        return os.path.join(self._bucketed_datasets_folder, f"{name}.parquet")

    @staticmethod
    def _build_name(pdf: PersistableDataFrame):
        return f"{pdf.name}_{pdf.uid}".replace('-', '__')


class CompositePersistenceManager(BasePersistenceManager):
    def __init__(self,
                 level2manager: Dict[PersistenceLevel, PersistenceManager],
                 parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)
        self._level2manager = level2manager

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        assert level in self._level2manager, \
            f"Cannot process level {level} because the corresponding manager has not been set. " \
            f"Only the following levels are supported: {list(self._level2manager.keys())}."

        persisted_on_levels = [
            lvl
            for lvl, manager in self._level2manager.items()
            if manager.is_persisted(pdf) and lvl != level
        ]

        assert len(persisted_on_levels) == 0, \
            f"Unable to persist with the required level {level}, because the dataset has been already persisted " \
            f"with different levels: {persisted_on_levels}"

        return self._level2manager[level].persist(pdf, level)

    def _unpersist(self, pdf: PersistableDataFrame):
        for lvl, manager in self._level2manager.items():
            if manager.is_persisted(pdf):
                manager.unpersist(pdf.uid)
                break

    def _create_child(self) -> PersistenceManager:
        level2managers = {lvl: manager.child() for lvl, manager in self._level2manager.items()}
        return CompositePersistenceManager(level2managers, self)
