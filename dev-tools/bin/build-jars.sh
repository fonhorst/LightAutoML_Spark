#!/usr/bin/env bash

set -e

cur_dir=$(pwd)

docker build -t lama-jar-builder -f docker/jar-builder/scala.dockerfile docker/jar-builder
docker run -it \
  -v "${cur_dir}/scala-lightautoml-transformers:/scala-lightautoml-transformers" \
  -v "${cur_dir}/jars:/jars" \
  lama-jar-builder
