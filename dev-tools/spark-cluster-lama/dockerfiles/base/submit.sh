#!/bin/bash

set -ex

script=$1

export SCRIPT_ENV=cluster
spark-submit \
  --master spark://node3.bdcl:7077 \
  --deploy-mode client \
  --conf 'spark.driver.host=node3.bdcl' \
  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
  --conf 'spark.kryoserializer.buffer.max=512m' \
  --conf 'spark.driver.cores=10' \
  --conf 'spark.driver.memory=20g' \
  --conf 'spark.executor.instances=4' \
  --conf 'spark.executor.cores=10' \
  --conf 'spark.executor.memory=20g' \
  --conf 'spark.cores.max=40' \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.cleaner.referenceTracking.cleanCheckpoints=true' \
  --conf 'spark.cleaner.referenceTracking=true' \
  --conf 'spark.cleaner.periodicGC.interval=1min' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --jars /opt/spark-lightautoml_2.12-0.1.jar \
  --py-files /opt/LightAutoML-0.3.0.tar.gz \
  ${script}
