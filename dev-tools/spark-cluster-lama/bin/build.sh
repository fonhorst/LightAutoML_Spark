#!/bin/bash

set -ex

docker build -t node2.bdcl:5000/spark:3.2.1-py3.9 -f dockerfiles/base/Dockerfile dockerfiles/base
docker push node2.bdcl:5000/spark:3.2.1-py3.9

