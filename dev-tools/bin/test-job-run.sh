#!/usr/bin/env bash

set -ex

job_name=$1
cfg_file="/tmp/${job_name}-config.yaml"
kube_ns="spark-lama-exps"

cat <<EOF > "${cfg_file}"
---
name: "some experiment"
params:
  param1: val1
  param2: val2
EOF

kubectl -n ${kube_ns} delete configmap "${job_name}-scripts" --ignore-not-found
kubectl -n ${kube_ns} create configmap "${job_name}-scripts" \
  --from-file=exec.py=dev-tools/performance_tests/spark_used_cars.py \
  --from-file=config.yaml="${cfg_file}"

kubectl -n ${kube_ns} delete job "${job_name}" --ignore-not-found
sed -e "s/{{jobname}}/${job_name}/g" dev-tools/config/spark-job.yaml.j2 | kubectl apply -f -

echo "Waiting for spark-job to complete..."
until \
  (kubectl -n ${kube_ns} get job/${job_name} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep True) \
  || (kubectl -n ${kube_ns} get job/${job_name} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep True); \
  do sleep 1 ; done

echo "Getting logs..."
kubectl -n ${kube_ns} logs job/${job_name}
