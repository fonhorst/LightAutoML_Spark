#!/usr/bin/env bash

set -e

kubectl -n spark-lama-exps delete configmap exp-test-scripts --ignore-not-found
kubectl -n spark-lama-exps create configmap exp-test-scripts --from-file=dev-tools/performance_tests/spark_used_cars.py
kubectl delete -f dev-tools/config/spark-job.yaml --ignore-not-found
kubectl apply -f dev-tools/config/spark-job.yaml


