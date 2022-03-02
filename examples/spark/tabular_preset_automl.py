import logging.config
import socket
from typing import Tuple

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, spark_session, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")  # .repartition(4)

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()
    # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


local_ip_address = socket.gethostbyname(socket.gethostname())


spark = (
    SparkSession
    .builder
    .master("k8s://https://10.32.15.3:6443")
    .config("spark.driver.bindAddress", '0.0.0.0')
    .config('spark.driver.host', local_ip_address)
    # .config('spark.kubernetes.container.image', 'node2.bdcl:5000/spark-py-lama:lama-v3.2.0')
    .config('spark.kubernetes.container.image', 'node2.bdcl:5000/spark-lama-k8s:3.9-3.2.0')
    .config('spark.kubernetes.namespace', 'spark-lama-exps')
    .config('spark.kubernetes.authenticate.driver.serviceAccountName', 'spark')
    .config('spark.kubernetes.memoryOverheadFactor', '0.4')
    .config('spark.kubernetes.driver.label.appname', 'driver-test-submit-run')
    .config('spark.kubernetes.executor.label.appname', 'executor-test-submit-run')
    .config('spark.kubernetes.executor.deleteOnTermination', 'false')
    .config('spark.scheduler.minRegisteredResourcesRatio', '1.0')
    .config('spark.scheduler.maxRegisteredResourcesWaitingTime', '180s')
    .config('spark.jars', 'jars/spark-lightautoml_2.12-0.1.jar')
    .config('spark.jars.packages', 'com.microsoft.azure:synapseml_2.12:0.9.5')
    .config('spark.jars.repositories', 'https://mmlspark.azureedge.net/maven')
    .config('spark.driver.cores', '4')
    .config('spark.driver.memory', '32g')
    .config('spark.executor.instances', '4')
    .config('spark.executor.cores', '4')
    .config('spark.executor.memory', '16g')
    .config('spark.cores.max', '16')
    .config('spark.memory.fraction', '0.6')
    .config('spark.memory.storageFraction', '0.5')
    .config('spark.sql.autoBroadcastJoinThreshold', '100MB')
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
    .config('spark.kubernetes.file.upload.path', '/mnt/nfs/spark_upload_dir')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.options.claimName', 'scripts-shared-vol')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.options.storageClass', 'local-hdd')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.mount.path', '/scripts/')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.mount.readOnly', 'true')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName', 'spark-lama-data')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass', 'local-hdd')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path', '/opt/spark_data/')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly', 'true')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName', 'spark-lama-data')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass', 'local-hdd')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path', '/opt/spark_data/')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly', 'true')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.claimName', 'exp-results-vol')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.storageClass', 'local-hdd')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.path', '/exp_results')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.readOnly', 'false')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.claimName', 'mnt-nfs')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass', 'nfs')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.path', '/mnt/nfs/')
    .config('spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly', 'false')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.claimName', 'mnt-nfs')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass', 'nfs')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.path', '/mnt/nfs/')
    .config('spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly', 'false')
    .getOrCreate()
)

seed = 42
cv = 5
use_algos = [["lgb"]]
path = "/opt/spark_data/sampled_app_train.csv"
task_type = "binary"
roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}

with log_exec_timer("spark-lama training") as train_timer:
    task = SparkTask(task_type)
    train_data, test_data = prepare_test_and_train(spark, path, seed)

    test_data_dropped = test_data

    automl = SparkTabularAutoML(
        spark=spark,
        task=task,
        general_params={"use_algos": use_algos},
        reader_params={"cv": cv, "advanced_roles": False},
        tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
    )

    oof_predictions = automl.fit_predict(
        train_data,
        roles=roles
    )

logger.info("Predicting on out of fold")

score = task.get_dataset_metric()
metric_value = score(oof_predictions)

logger.info(f"score for out-of-fold predictions: {metric_value}")

with log_exec_timer("spark-lama predicting on test") as predict_timer:
    te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

    score = task.get_dataset_metric()
    test_metric_value = score(te_pred)

    logger.info(f"score for test predictions: {test_metric_value}")

logger.info("Predicting is finished")

result = {
    "metric_value": metric_value,
    "test_metric_value": test_metric_value,
    "train_duration_secs": train_timer.duration,
    "predict_duration_secs": predict_timer.duration
}

print(f"EXP-RESULT: {result}")

spark.stop()