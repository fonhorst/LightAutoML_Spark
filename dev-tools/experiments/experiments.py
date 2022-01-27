import os.path
import socket
import itertools
import argparse

from test import complete_experiment
from multiprocessing import Pool

LOCAL_IP = socket.gethostbyname(socket.gethostname())

parser = argparse.ArgumentParser(description="LAMA Spark experiments")

# Permutations arguments
parser.add_argument(
    "--spark_executor_num", type=str, help='Number of spark executors - input format "3,4,5"', default="2,3"
)
parser.add_argument(
    "--spark_executor_memory", type=str, help='Spark executors memory in Gb - input format "1,2,3" ', default="1"
)
parser.add_argument("--dataset_size", type=str, help='Dataset size - input format "100,1000,3000000" ', default="100")

# Service arguments
parser.add_argument(
    "--spark_master",
    type=str,
    help="Type of Spark master - k8s or local (default local[1]) ",
    default="k8s://https://192.168.137.152:8443",
)  # default = "k8s://https://192.168.137.152:8443"
parser.add_argument("--use_state_file", type=str, help="Usage of state file - [use|ignore] ", default="ignore")

parser.add_argument(
    "--spark_kubernetes_authenticate_serviceAccountName", type=str, help="K8s service account ", default="spark"
)
parser.add_argument("--spark_kubernetes_namespace", type=str, help="K8s namespace ", default="default")
parser.add_argument(
    "--spark_kubernetes_container_image", type=str, help="Container image for K8s pod ", default="local/spark-lama"
)
parser.add_argument("--spark_driver_host", type=str, help="Driver host IP", default=LOCAL_IP)
parser.add_argument("--spark_driver_bindAddress", type=str, help="Driver bind adress", default="0.0.0.0")

# parser.add_argument('--spark_kubernetes_authenticate_submission_caCertFile', type=str, help='Certificate location', default = "/root/.minikube/ca.crt")
# parser.add_argument('--spark_submit_deployMode', type=str, help='Deploy mode', default = "cluster") -- K8s doesn`t support cluster mode from SparkSession?

arg_space = parser.parse_args()

spark_executor_num = list(map(int, arg_space.spark_executor_num.split(sep=",")))
spark_executor_memory = list(map(int, arg_space.spark_executor_memory.split(sep=",")))
dataset_size = list(map(int, arg_space.dataset_size.split(sep=",")))

arg_product = list(itertools.product(spark_executor_num, spark_executor_memory, dataset_size))

if __name__ == "__main__":
    with Pool(processes=len(arg_product)) as pool:

        expeiments_dict = {}
        session_args = []

        if arg_space.use_state_file in ("use"):
            # Check experiment sate file and put them into dictionary
            if os.path.exists("./dev-tools/experiments/experiment.log"):
                with open("./dev-tools/experiments/experiment.log", "r") as file:
                    lines = file.readlines()

                for item in lines:
                    vals = item.split(sep=":")
                    expeiments_dict[vals[0]] = float(vals[1])

        elif arg_space.use_state_file == "ignore":
            # Delete existing state file
            open(f"./dev-tools/experiments/experiment.log", "w+").close()

        for item in arg_product:
            experiment_name = f"Experiment {item[0]}_{item[1]}_{item[2]}"
            if experiment_name in expeiments_dict:
                continue

            arg_dict = {}
            for key in vars(arg_space):
                if key not in (
                    "appName",
                    "spark_executor_instances",
                    "spark_executor_memory",
                    "spark_master",
                    "dataset_size",
                    "use_state_file",
                ):
                    arg_dict[key.replace("_", ".")] = vars(arg_space)[key]

            spark_args = dict(
                {
                    "appName": f"Experiment {item[0]}_{item[1]}_{item[2]}",
                    "spark.executor.instances": item[0],
                    "spark.executor.memory": f"{item[1]}g",
                },
                **arg_dict,
            )

            session_args.append([experiment_name, int(item[2]), spark_args, arg_space.spark_master])  # Data size

        # Run experiments
        experiments_map = pool.starmap(complete_experiment, session_args)

        for item in experiments_map:
            for key in item:
                if key not in expeiments_dict:
                    expeiments_dict[key] = item[key]

        # Write experiment results into file
        with open(f"./dev-tools/experiments/experiment.log", "w+") as file:
            for key in expeiments_dict:
                file.write(f"{key}:{expeiments_dict[key]}\n")

        print(expeiments_dict)
