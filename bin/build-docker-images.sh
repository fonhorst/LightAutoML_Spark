#!/usr/bin/env bash

set -e

./bin/build-jars.sh

# shellcheck disable=SC2094
poetry export -f requirements.txt > requirements.txt
poetry build

docker build \
  -t spark-py-lama:lama-v3.2.0 \
  -f docker/spark-lama/spark-py-lama.dockerfile \
  .

rm -rf dist
