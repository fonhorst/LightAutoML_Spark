#!/bin/bash

set -ex

base_dir="/mnt/nfs/scluster"
compose_dir="${base_dir}/spark-cluster-lama"
control_node="node3.bdcl"
master="node3.bdcl"
workers=()

for (( node=4; node<=8; node++ ))
do
	workers+=("node${node}.bdcl")
done

function create_config_on_control_node() {
  ssh ${control_node} "mkdir -p ${base_dir} && rm -rf ${base_dir}/spark-cluster-lama"
  scp -r dev-tools/spark-cluster-lama ${control_node}:${compose_dir}
}

function deploy_scluster() {
  local force_mode="${1:-false}"    # Default value is false

  if "${force_mode}"; then
      args="--force-recreate"
  else
      args=""
  fi

  echo "Starting the master on host: ${master}"
  # shellcheck disable=SC2029
  ssh ${master} "cd ${compose_dir} && docker-compose up -d ${args} spark-master spark-submit"
  
  for host in "${workers[@]}"
  do
      echo "Starting a worker on host: ${host}"
      # shellcheck disable=SC2029
      ssh $host "cd ${compose_dir} && docker-compose up -d ${args} spark-worker"
  done
}

function teardown_scluster() {
  for host in "${workers[@]}"
  do
      echo "Stopping a worker on host: ${host}"
      # shellcheck disable=SC2029
      ssh $host "cd ${compose_dir} && docker-compose down spark-worker"
  done     
  
  echo "Stopping the master on host: ${master}"
  # shellcheck disable=SC2029
  ssh ${master} "cd ${compose_dir} && docker-compose down spark-master spark-submit"
}

function help() {
  echo "
  List of commands.
    create-config -  create docker-compose file remotely on control node (${control_node})
    deploy - deploy spark cluster (master will be on ${master})
    redeploy - forcefully deploy spark cluster
    teardown - stop spark cluster
    help - prints this message
  "
}


function main () {
    cmd="$1"

    if [ -z "${cmd}" ]
    then
      echo "No command is provided."
      help
      exit 1
    fi

    shift 1

    echo "Executing command: ${cmd}"

    case "${cmd}" in
    "create-config")
        create_config_on_control_node
        ;;

    "deploy")
        deploy_scluster false
        ;;

    "redeploy")
        deploy_scluster true
        ;;

    "teardown")
        teardown_scluster
        ;;

    "help")
        help
        ;;

    *)
        echo "Unknown command: ${cmd}"
        ;;

    esac
}

main "${@}"
