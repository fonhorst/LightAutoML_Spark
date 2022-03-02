#!/usr/bin/env bash

set -e

#./bin/build-jars.sh

repo="node2.bdcl:5000"

# shellcheck disable=SC2094
poetry export -f requirements.txt > requirements.txt
poetry build

docker build \
  -t ${repo}/spark-py-lama:lama-v3.2.0 \
  -f docker/spark-lama/spark-py-lama.dockerfile \
  .

rm -rf dist
