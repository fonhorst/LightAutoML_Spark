import logging.config
import os
from typing import List

from lightautoml.dataset.roles import NumericRole
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import BucketedPersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, SparkDataFrame
from .. import spark as spark_sess, BUCKET_NUMS
import numpy as np

spark = spark_sess

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

        in_cols = ["id", "fold", "seed", "a", "b", "c"]

        roles = {col: NumericRole(np.int32) for col in in_cols}

        df_data = [
            {col: val for col, val in zip(in_cols, row)}
            for row in data
        ]
        df = spark.createDataFrame(df_data)#, schema=schema)

        return SparkDataset(df, roles, name=name)

    os.environ["HADOOP_HOME"] = "/etc/hadoop-bdcl/"

    pmanager = BucketedPersistenceManager(
        bucketed_datasets_folder="hdfs:///tmp/slama_test_workdir",
        bucket_nums=BUCKET_NUMS
    )

    ds_1 = pmanager.persist(make_ds("dataset_1"))
    ds_2 = pmanager.persist(make_ds("dataset_2"))
    pmanager.unpersist(ds_1.uid)

    pmanager.unpersist_all()
