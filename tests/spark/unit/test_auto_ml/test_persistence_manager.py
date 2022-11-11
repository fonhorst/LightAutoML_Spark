import logging.config
import os

import numpy as np
from lightautoml.dataset.roles import NumericRole
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import BucketedPersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from .. import BUCKET_NUMS, spark_hdfs, HDFS_TMP_SLAMA_DIR

spark = spark_hdfs

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def test_bucketed_persistence_manager(spark: SparkSession):
    def make_ds(name: str) -> SparkDataset:
        data = [
            [0, 0, 42, 1, 1, 1],
            [1, 0, 43, 2, 1, 3],
            [2, 1, 44, 1, 2, 3],
            [3, 1, 45, 1, 2, 2],
            [4, 2, 46, 3, 1, 1],
            [5, 2, 47, 4, 1, 2],
        ]

        in_cols = ["_id", "fold", "seed", "a", "b", "c"]

        roles = {col: NumericRole(np.int32) for col in in_cols}

        df_data = [
            {col: val for col, val in zip(in_cols, row)}
            for row in data
        ]
        df = spark.createDataFrame(df_data)

        return SparkDataset(df, roles, name=name)

    os.environ["HADOOP_HOME"] = "/etc/hadoop-bdcl/"

    pmanager = BucketedPersistenceManager(
        bucketed_datasets_folder=f"hdfs://node21.bdcl:9000{HDFS_TMP_SLAMA_DIR}",
        bucket_nums=BUCKET_NUMS
    )

    ds_1 = pmanager.persist(make_ds("dataset_1"))
    ds_2 = pmanager.persist(make_ds("dataset_2"))
    pmanager.unpersist(ds_1.uid)

    pmanager.unpersist_all()
