import os
from typing import Optional

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, PersistenceLevel, PersistableDataFrame, PersistenceManager


# TODO: SLAMA - add documentation


class PlainCachePersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        ds = SparkSession.getActiveSession().createDataFrame(pdf.sdf.rdd, schema=pdf.sdf.schema).cache()
        ds.write.mode('overwrite').format('noop').save()

        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class LocalCheckpointPersistenceManager(PersistenceManager):
    def __init__(self, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
        ds = pdf.sdf.localCheckpoint()
        return PersistableDataFrame(ds, pdf.uid, pdf.callback, pdf.base_dataset)

    def _unpersist(self, pdf: PersistableDataFrame):
        pdf.sdf.unpersist()


class BucketedPersistenceManager(PersistenceManager):
    def __init__(self, bucketed_datasets_folder: str, bucket_nums: int = 100, parent: Optional['PersistenceManager'] = None):
        super().__init__(parent)
        self._bucket_nums = bucket_nums
        self._bucketed_datasets_folder = bucketed_datasets_folder

    def _persist(self, pdf: PersistableDataFrame, level: PersistenceLevel) -> PersistableDataFrame:
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
