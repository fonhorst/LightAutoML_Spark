#!/bin/bash

set -ex

BASE_IMAGE_NAME=node2.bdcl:5000/spark-base:3.2.1-py3.9
MASTER_WORKER_IMAGE_NAME=node2.bdcl:5000/spark-master-worker:3.2.1-py3.9
SUBMITTER_IMAGE_NAME=node2.bdcl:5000/spark-submitter:3.2.1-py3.9

DOCKERFILE_FOLDER=dev-tools/spark-cluster-lama/dockerfiles/base
BUILD_FOLDER=dev-tools/spark-cluster-lama/dockerfiles/base
BUILD_TMP_FOLDER=${BUILD_FOLDER}/build_tmp

poetry export -f requirements.txt > requirements.txt
poetry build

rm -rf ${BUILD_TMP_FOLDER}
mkdir ${BUILD_TMP_FOLDER}
cp requirements.txt ${BUILD_TMP_FOLDER}
cp -r dist ${BUILD_TMP_FOLDER}/dist
cp jars/spark-lightautoml_2.12-0.1.jar ${BUILD_TMP_FOLDER}/spark-lightautoml_2.12-0.1.jar

docker build -t ${BASE_IMAGE_NAME} -f ${DOCKERFILE_FOLDER}/base.dockerfile ${BUILD_FOLDER}
docker push ${BASE_IMAGE_NAME}

docker build -t ${MASTER_WORKER_IMAGE_NAME} -f ${DOCKERFILE_FOLDER}/spark-master-worker.dockerfile ${BUILD_FOLDER}
docker push ${MASTER_WORKER_IMAGE_NAME}

docker build -t ${SUBMITTER_IMAGE_NAME} -f ${DOCKERFILE_FOLDER}/spark-submitter.dockerfile ${BUILD_FOLDER}
docker push ${SUBMITTER_IMAGE_NAME}

rm -rf ${BUILD_TMP_FOLDER}
