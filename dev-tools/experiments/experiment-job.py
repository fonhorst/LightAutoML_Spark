import itertools
import yaml
import subprocess

with open("./dev-tools/experiments/experiment-config.yaml", "r") as stream:
    try:
        config_data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

repeat_rate = config_data["spark_quality_repeat_rate"]

# Make all possible experiments
keys_exps, values_exps = zip(*config_data["experiment_params"].items())
experiments = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

# Retrieve static Spark paramaters and concatenate them with experiment paramaters
experiments_configs = []
for item in experiments:
    t_dict = dict(item)
    t_dict["use_state_file"] = config_data["use_state_file"]
    t_dict.update(config_data["spark_static_config"])
    experiments_configs.append(t_dict)

# Run subprocesses with Spark jobs
for experiment in experiments_configs:
    processes = []
    for i in range(repeat_rate):
        name = (
            f"exper-{experiment['dataset_path'].translate(str.maketrans({'/':'-', '_':'-', '.':'-'}))}"
            f"-{experiment['spark.executor.instances']}-{experiment['spark.executor.memory']}-n{i+1}"
        )
        experiment["name"] = name

        with open(f"/tmp/{name}-config.yaml", "w+") as outfile:
            yaml.dump(experiment, outfile, default_flow_style=False)

        processes.append(subprocess.Popen(["./dev-tools/bin/test-job-run.sh", str(name)]))

    for p in processes:
        p.wait()

print("Finished processes")
