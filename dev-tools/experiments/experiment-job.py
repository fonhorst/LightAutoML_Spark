import itertools
import os
import subprocess
import time
import json
import re
from datetime import datetime
from copy import deepcopy, copy
from typing import Iterable, List, Set, Dict, Any, Iterator, Tuple, Optional

import yaml
from tqdm import tqdm

patterns = [r"Test results:\[.*\]"]
statefile_path = "./dev-tools/experiments/results"
results_path = "./dev-tools/experiments/results"
cfg_path = "./dev-tools/config/experiments/test-experiment-config.yaml"


ExpInstanceConfig = Dict[str, Any]


def read_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    return config_data


def process_state_file(config_data: Dict[str, Any]) -> Set[str]:
    state_file_mode = config_data["state_file"]
    state_file_path = f"{statefile_path}/state_file.json"

    if state_file_mode == "use":
        with open(state_file_path, "r") as f:
            exp_instances = [json.loads(line) for line in  f.readlines()]
        exp_instances_ids = {exp_inst["instance_id"] for exp_inst in exp_instances}
    elif state_file_mode == "delete":
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
        exp_instances_ids = set()
    else:
        raise ValueError(f"Unsupported mode for state file: {state_file_mode}")

    return exp_instances_ids


def generate_experiments(config_data: Dict) -> List[ExpInstanceConfig]:
    experiments = config_data["experiments"]

    existing_exp_instances_ids = process_state_file(config_data)

    exp_instances = []
    for experiment in experiments:
        name = experiment["name"]
        repeat_rate = experiment["repeat_rate"]
        libraries = experiment["library"]

        # Make all possible experiment AutoML params
        keys_exps, values_exps = zip(*experiment["params"].items())
        param_sets = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

        if "spark" in libraries:
            assert "spark_config" in experiment, f"No spark_config set (even empty one) for experiment {name}"
            keys_exps, values_exps = zip(*experiment['params']['spark_config'].items())
            spark_param_sets = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

            spark_configs = []
            for spark_params in spark_param_sets:
                spark_config = copy(config_data["default_spark_config"])
                spark_config.update(spark_params)
                spark_config['spark.cores.max'] = \
                    int(spark_config['spark.executor.cores']) * int(spark_config['spark.executor.instances'])
                spark_configs.append(spark_config)

        else:
            spark_configs = []

        for library, params, spark_config in itertools.product(libraries, param_sets, spark_configs):
            params = copy(params)

            use_algos = '__'.join(['_'.join(layer) for layer in params['use_algos']])

            instance_id = f"{name}-{params['dataset']}-{use_algos}-{params['cv']}-{params['seed']}-" \
                   f"{params['spark_config']['spark.executor.instances']}"

            if instance_id in existing_exp_instances_ids:
                continue

            if library == "spark":
                params["spark_config"] = spark_config

            exp_instance = {
                "exp_name": name,
                "instance_id": instance_id,
                "params": params,
                "calculation_script": config_data["calculation_script"]
            }

            exp_instances.append(exp_instance)

    return exp_instances


# Run subprocesses with Spark jobs
def run_experiments(experiments_configs: List[ExpInstanceConfig]) \
        -> Iterator[Tuple[ExpInstanceConfig, subprocess.Popen]]:
    for exp_instance in experiments_configs:
        instance_id = exp_instance["instance_id"]
        launch_script_name = exp_instance["calculation_script"]
        with open(f"/tmp/{instance_id}-config.yaml", "w+") as outfile:
            yaml.dump(exp_instance["experiment_params"], outfile, default_flow_style=False)

        with open(f"{results_path}/Results_{instance_id}.log", "w+") as logfile:
            logfile.write(f"Launch datetime: {datetime.now()}\n")
            # p = subprocess.Popen(["./dev-tools/bin/test-sleep-job.sh", str(name)], stdout=logfile)
            p = subprocess.Popen(
                ["./dev-tools/bin/test-job-run.sh", instance_id, str(launch_script_name)],
                stdout=logfile
            )

        print(f"Starting exp with name {instance_id}")
        yield exp_instance, p


def limit_procs(it: Iterator[Tuple[ExpInstanceConfig, subprocess.Popen]],
                max_parallel_ops: int = 1,
                check_period_secs: float = 1.0) \
        -> Iterator[Tuple[ExpInstanceConfig, subprocess.Popen]]:
    assert max_parallel_ops > 0

    exp_procs: Set[Tuple[ExpInstanceConfig, subprocess.Popen]] = set()

    def try_to_remove_finished() -> Optional[Tuple[ExpInstanceConfig, subprocess.Popen]]:
        proc_to_remove = None
        for exp_instance, p in exp_procs:
            if p.poll() is not None:
                proc_to_remove = (exp_instance, p)
                break

        if proc_to_remove is not None:
            exp_procs.remove(proc_to_remove)
            return proc_to_remove

        return None

    for el in it:
        exp_procs.add(el)

        while len(exp_procs) >= max_parallel_ops:
            exp_proc = try_to_remove_finished()

            if exp_proc:
                yield exp_proc
            else:
                time.sleep(check_period_secs)

    while len(exp_procs) > 0:
        exp_proc = try_to_remove_finished()

        if exp_proc:
            yield exp_proc
        else:
            time.sleep(check_period_secs)


def wait_for_all(exp_procs: Iterator[Tuple[ExpInstanceConfig, subprocess.Popen]], total: int):
    state_file_path = f"{statefile_path}/state_file.json"

    for exp_instance, p in tqdm(exp_procs, desc="Experiment", total=total):
        # Mark exp_instance as completed in the state file
        with open(state_file_path, "a") as f:
            record = json.dumps(exp_instance)
            f.write(f"{record}{os.linesep}")


def gather_results(procs: Iterable[subprocess.Popen]):
    with open(f"{results_path}/Experiments_results.txt", "a+") as resultfile:
        for p in procs:
            with open(f"{results_path}/Results_{p.args[1]}.log", "r") as logfile:
                stdout = logfile.read()

                for pattern in patterns:
                    metrics = re.search(pattern, stdout)
                    if metrics:
                        metrics = metrics.group()
                        metrics = metrics.split("[", 1)[1].split("]")[0]
                        break

                print(f"Obtained results for experiment {p.args[1]}: {metrics}")
                resultfile.write(f"{p.args[1]}:{metrics}\n")
                print(f"Results for experiment {p.args[1]} saved in file\n")


def main():
    cfg = read_config(cfg_path)
    exp_cfgs = generate_experiments(cfg)
    exp_procs = limit_procs(run_experiments(exp_cfgs), max_parallel_ops=2)
    wait_for_all(exp_procs, total=len(exp_cfgs))
    gather_results(exp_procs)

    print("Finished processes")


if __name__ == "__main__":
    main()
