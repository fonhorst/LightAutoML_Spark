import logging
import os
import shutil
import uuid
from abc import abstractmethod
from typing import Optional, Dict, List, Union, cast

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, PersistenceLevel, PersistableDataFrame, PersistenceManager

logger = logging.getLogger(__name__)


class BasePersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        self._uid = str(uuid.uuid4())
        self._persistence_registry: Dict[str, PersistableDataFrame] = dict()
        self._parent = parent
        self._children: List['PersistenceManager'] = []

    @property
    def uid(self) -> str:
        return self._uid

    def persist(self,
                dataset: Union[SparkDataset, PersistableDataFrame],
                level: PersistenceLevel = PersistenceLevel.REGULAR) -> PersistableDataFrame:
        persisted_dataframe = self.to_persistable_dataframe(dataset) if isinstance(dataset, SparkDataset) \
            else cast(PersistableDataFrame, dataset)

        logger.info(f"Manager {self._uid}: "
                    f"persisting dataset (uid={dataset.uid}, name={dataset.name}) with level {level}.")

        if persisted_dataframe.uid in self._persistence_registry:
            logger.debug(f"Manager {self._uid}: "
                         f"the dataset (uid={dataset.uid}, name={dataset.name}) is already persisted.")
            return self._persistence_registry[persisted_dataframe.uid]

        self._persistence_registry[persisted_dataframe.uid] = self._persist(persisted_dataframe, level)

        logger.debug(f"Manager {self._uid}: the dataset (uid={dataset.uid}, name={dataset.name}) has been persisted.")

        return persisted_dataframe

    def unpersist(self, uid: str):
        logger.info(f"Manager {self._uid}: unpersisting dataset (uid={uid}).")
        persisted_dataframe = self._persistence_registry.get(uid, None)

        if not persisted_dataframe:
            logger.debug(f"Manager {self._uid}: the dataset (uid={uid}) is not persisted yet. Nothing to do.")
            return

        self._unpersist(persisted_dataframe)

        del self._persistence_registry[persisted_dataframe.uid]

        logger.debug(f"Manager {self._uid}: the dataset (uid={uid}, name={persisted_dataframe.name}) has been unpersisted.")

    def unpersist_children(self):
        logger.info(f"Manager {self._uid}: unpersisting children.")

        for child in self._children:
            child.unpersist_all()
        self._children = []

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
        a_child = PersistenceManager(self)
        self._children.append(a_child)
        logger.info(f"Manager {self._uid}: the child (uid={a_child.uid}) has been produced.")
        return a_child

    def is_persisted(self, pdf: PersistableDataFrame) -> bool:
        return pdf.uid in self._persistence_registry

    @abstractmethod
    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        ...

    @abstractmethod
    def _unpersist(self, pdf: PersistableDataFrame):
        ...


class PlainCachePersistenceManager(BasePersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        logger.debug(f"Manager {self._uid}: "
                     f"caching and materializing the dataset (uid={pdf.uid}, name={pdf.name}).")

        ds = SparkSession.getActiveSession().createDataFrame(pdf.sdf.rdd, schema=pdf.sdf.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        logger.debug(f"Manager {self._uid}: "
                     f"caching succeeded for the dataset (uid={pdf.uid}, name={pdf.name}).")

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class LocalCheckpointPersistenceManager(BasePersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        logger.debug(f"Manager {self._uid}: "
                     f"making a local checkpoint for the dataset (uid={pdf.uid}, name={pdf.name}).")

        ds = pdf.sdf.localCheckpoint()

        logger.debug(f"Manager {self._uid}: "
                     f"the local checkpoint has been made for the dataset (uid={pdf.uid}, name={pdf.name}).")
        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class BucketedPersistenceManager(BasePersistenceManager):
    def __init__(self,
                 bucketed_datasets_folder: str,
                 bucket_nums: int = 100,
                 parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)
        self._bucket_nums = bucket_nums
        self._bucketed_datasets_folder = bucketed_datasets_folder

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        spark = SparkSession.getActiveSession()
        name = self._build_name(pdf)
        # TODO: SLAMA join - need to identify correct setting  for bucket_nums if it is not provided
        path = self._build_path(name)
        logger.debug(
            f"Manager {self._uid}: making a bucketed table "
            f"for the dataset (uid={pdf.uid}, name={pdf.name}) with name {name} on path {path}."
        )

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

    def _build_path(self, name: str) -> str:
        return os.path.join(self._bucketed_datasets_folder, f"{name}.parquet")

    @staticmethod
    def _build_name(pdf: PersistableDataFrame):
        return f"{pdf.name}_{pdf.uid}"


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
