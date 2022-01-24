#!/usr/bin/env bash

set -e

APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')


# TODO: volumes
# TODO: memory
# TODO: num_executors

spark-submit \
  --master k8s://${APISERVER} \
  --deploy-mode cluster \
  --conf 'spark.kubernetes.container.image=node2.bdcl:5000/spark-lama-k8s:3.9-3.2.0' \
  --conf 'spark.kubernetes.container.image.pullPolicy=Always' \
  --conf 'spark.kubernetes.memoryOverheadFactor=0.1' \
  --conf 'spark.driver.cores=4' \
  --conf 'spark.executor.cores=4' \
  --conf 'spark.cores.max=16' \
  --conf 'spark.memory.fraction= 0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  dev-tools/performance_tests/spark_used_cars.py