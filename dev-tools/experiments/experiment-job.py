import itertools
import subprocess
import time
import ast
from copy import deepcopy
from typing import Iterable, List, Set, Dict, Any

import yaml
from tqdm import tqdm


def read_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    return config_data


def generate_experiments(config_data: Dict) -> List[Dict[str, Any]]:
    repeat_rate = config_data["spark_quality_repeat_rate"]
    # Make all possible experiments
    keys_exps, values_exps = zip(*config_data["experiment_params"].items())
    experiments = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

    # Process state file if neccessary
    use_state_file = config_data["use_state_file"]
    exps_to_skip = []

    if use_state_file == "use":
        with open("./dev-tools/experiments/results/state_file.txt", "r") as file:
            exps_to_skip = file.read().split(",")

    elif use_state_file == "delete":
        open("./dev-tools/experiments/results/state_file.txt", "w").close()

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
            if name in exps_to_skip:
                continue

            exp_instance = deepcopy(experiment)
            exp_instance["name"] = name
            experiments_configs.append(exp_instance)

    return experiments_configs


# Run subprocesses with Spark jobs
def run_experiments(experiments_configs: List[Dict[str, Any]]):
    for experiment in experiments_configs:
        name = experiment["name"]
        with open(f"/tmp/{name}-config.yaml", "w+") as outfile:
            yaml.dump(experiment, outfile, default_flow_style=False)

        # p = subprocess.Popen(["./dev-tools/bin/test-sleep-job.sh", str(name)], stdout=subprocess.PIPE)
        p = subprocess.Popen(["./dev-tools/bin/test-job-run.sh", str(name)], stdout=subprocess.PIPE)
        print(f"Starting exp with name {name}")
        yield p


def limit_procs(it: Iterable[subprocess.Popen], max_parallel_ops: int = 1):
    assert max_parallel_ops > 0

    procs: Set[subprocess.Popen] = set()

    def try_to_remove_finished():
        for p in procs:
            if p.poll() is not None:
                procs.remove(p)
                break

    for el in it:
        procs.add(el)
        yield el

        # while len(procs) >= max_parallel_ops:  # TODO: fix processes limits
        #     try_to_remove_finished()
        #     time.sleep(1)


def wait_for_all(procs: Iterable[subprocess.Popen]) -> Dict[str, Any]:
    results = {}
    for p in procs:
        stdout = p.communicate()

        # Mark task completion in state file
        with open("./dev-tools/experiments/results/state_file.txt", "a+") as file:
            file.write(f"{p.args[1]},")

        # Print stdout and save results
        stdout = stdout[0].decode("utf-8")
        print(f"\n------- STDOUT of experiment {p.args[1]} -------\n{stdout}")

        results[p.args[1]] = stdout.splitlines()[-1]

    return results


def write_results(results: Dict[str, Any]):
    with open("./dev-tools/experiments/results/Experiments_results.txt", "a+") as file:
        for key, value in results.items():
            # Check if acquired results are correct and write them
            try:
                ast.literal_eval(value)
            except Exception as ex:
                print(f"Obtained results for experiment {key} are incorrect")
                print(f"Input results: {value}\nError message: {ex}")
                continue

            file.write(f"{key}:{value}\n")
            print(f"Results for experiment {key} saved in file")


def main():
    cfg_path = "./dev-tools/config/experiments/test-experiment-config.yaml"
    cfg = read_config(cfg_path)
    exp_cfgs = generate_experiments(cfg)
    exp_procs = list(
        tqdm(limit_procs(run_experiments(exp_cfgs), max_parallel_ops=2), desc="Experiment", total=len(exp_cfgs))
    )
    results = wait_for_all(exp_procs)
    write_results(results)

    print("Finished processes")


if __name__ == "__main__":
    main()
