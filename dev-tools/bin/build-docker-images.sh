#!/usr/bin/env bash

set -e

docker build -t spark-pyspark-python:3.9-3.2.0 -f dev-tools/docker/spark-pyspark-python.dockerfile .

# shellcheck disable=SC2094
poetry export -f requirements.txt > requirements.txt
poetry build
cp -r "${HOME}/.ivy2/cache" jars_cache
docker build \
  -t spark-lama:3.9-3.2.0 \
  -f dev-tools/docker/spark-lama.dockerfile \
  .

docker build -t spark-lama-k8s:3.9-3.2.0 -f dev-tools/docker/spark-lama-k8s.dockerfile .

rm -rf jars_cache
rm -rf dist
