#!/usr/bin/env bash

set -e

repo=node2.bdcl:5000

export SPARK_VERSION=3.2.0
export HADOOP_VERSION=3.2

mkdir -p /tmp/spark-build-dir
cd /tmp/spark-build-dir

wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
  && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# create images with names:
# - ${repo}/spark:lama-v3.2.0
# - ${repo}/spark-py:lama-v3.2.0
./spark/bin/docker-image-tool.sh -r "${repo}" -t lama-v3.2.0 \
  -p spark/kubernetes/dockerfiles/spark/bindings/python/Dockerfile \
  build

./spark/bin/docker-image-tool.sh -r "${repo}" -t lama-v3.2.0 push
