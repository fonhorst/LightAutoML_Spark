#!/usr/bin/env bash

set -e

APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

KUBE_NAMESPACE=spark-lama-exp

remote_script_path=run.py
scp examples/spark/tabular_preset_automl.py \
  node2.bdcl:/mnt/ess_storage/DN_1/tmp/scripts-shared-vol/${remote_script_path}

spark-submit \
  --master k8s://${APISERVER} \
  --deploy-mode cluster \
  --conf 'spark.driver.bindAddress=0.0.0.0' \
  --conf 'spark.kubernetes.namespace='${KUBE_NAMESPACE} \
  --conf 'spark.kubernetes.authenticate.driver.serviceAccountName=spark' \
  --conf 'spark.kubernetes.memoryOverheadFactor=0.4' \
  --conf 'spark.kubernetes.driver.label.appname=driver-test-submit-run' \
  --conf 'spark.kubernetes.executor.label.appname=executor-test-submit-run' \
  --conf 'spark.kubernetes.executor.deleteOnTermination=false' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --conf 'spark.jars=/jars/spark-lightautoml_2.12-0.1.jar' \
  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
  --conf 'spark.driver.cores=4' \
  --conf 'spark.driver.memory=32g' \
  --conf 'spark.executor.instances=4' \
  --conf 'spark.executor.cores=4' \
  --conf 'spark.executor.memory=16g' \
  --conf 'spark.cores.max=16' \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.claimName=exp-results-vol' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.path=/exp_results' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.readOnly=false' \
  local:///scripts/${remote_script_path}
