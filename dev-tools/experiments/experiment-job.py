import itertools
import subprocess
import time
from copy import deepcopy
from typing import Iterable, List, Set, Dict, Any

import yaml
from tqdm import tqdm


def read_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config_data


def generate_experiments(config_data: Dict) -> List[Dict[str, Any]]:
    repeat_rate = config_data["spark_quality_repeat_rate"]
    # Make all possible experiments
    keys_exps, values_exps = zip(*config_data["experiment_params"].items())
    experiments = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

    # Retrieve static Spark paramaters and concatenate them with experiment paramaters
    experiments_configs = []
    for item in experiments:
        experiment = dict(item)
        experiment["use_state_file"] = config_data["use_state_file"]
        experiment.update(config_data["spark_static_config"])
        for i in range(repeat_rate):
            name = (
                f"exper-{experiment['dataset_path'].translate(str.maketrans({'/': '-', '_': '-', '.': '-'}))}"
                f"-{experiment['spark.executor.instances']}-{experiment['spark.executor.memory']}-n{i + 1}"
            )
            exp_instance = deepcopy(experiment)
            exp_instance["name"] = name
            experiments_configs.append(exp_instance)

    return experiments_configs


# Run subprocesses with Spark jobs
# for i in range(repeat_rate):
def run_experiments(experiments_configs: List[Dict[str, Any]]):
    processes = []
    for experiment in experiments_configs:
        name = experiment["name"]
        with open(f"/tmp/{name}-config.yaml", "w+") as outfile:
            yaml.dump(experiment, outfile, default_flow_style=False)

        p = subprocess.Popen(["./dev-tools/bin/test-job-run.sh", str(name)])
        yield p

    for p in processes:
        p.wait()


def limit_procs(it: Iterable[subprocess.Popen], max_parallel_ops: int = 1):
    assert max_parallel_ops > 0

    procs: Set[subprocess.Popen] = set()

    def try_to_remove_finished():
        for p in procs:
            if p.poll():
                procs.remove(p)
                break

    for el in it:
        procs.add(el)
        yield el

        while len(procs) >= max_parallel_ops:
            try_to_remove_finished()
            time.sleep(5)


def wait_for_all(procs: Iterable[subprocess.Popen]):
    for p in procs:
        p.wait()


def main():
    cfg_path = "./dev-tools/experiments/experiment-config.yaml"
    cfg = read_config(cfg_path)
    exp_cfgs = generate_experiments(cfg)
    exp_procs = list(tqdm(
        limit_procs(run_experiments(exp_cfgs), max_parallel_ops=2),
        desc="Experiment",
        total=len(exp_cfgs)
    ))
    wait_for_all(exp_procs)
    print("Finished processes")


if __name__ == "__main__":
    main()
