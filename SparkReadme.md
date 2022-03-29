Examples for Spark-LAMA can be found in examples/spark/.
These examples can be run both locally and remotely on a cluster.

To run examples locally one needs just ensure that data files lay in appropriate locations.
These locations typically /opt/spark_data directory.
(Data for the examples can be found in examples/data)

To run examples remotely on a cluster under Kubernetes control one needs 
to have installed and configured **kubectl** utility.
#### 1. Establish nfs / S3 
This step is necessary to make uploading of script file  
(e.g. executable of Spark LAMA) into a location that is accessible from anywhere on cluster.
This file will be used by spark driver which is also submitted to the cluster.
Upon configuring set appropriate value for *spark.kubernetes.file.upload.path* in ./bin/slamactl.sh or mount it to /mnt/nfs on the localhost.

#### 2. Create persistent volumes (PV) and claims (PVC)

Examples required 2 PVC for their functioning (defined in slamactl.sh, spark-submit arguments):
 - *spark-lama-data* - provides access for driver and executors to data
 - *mnt-nfs* - provide access for driver and executors to the mentioned above upload dir

#### 3. Define required env variables
Define required environment variables to use appropriate kubernetes namespace 
and remote docker repository accessible from anywhere in the cluster.
```shell
export KUBE_NAMESPACE=spark-lama-exps 
export REPO=node2.bdcl:5000 
```

#### 4. Build spark lama dependencies and docker images.
On this step use slamactl.sh utility to build dependencies and docker images:
```shell
./bin/slamactl.sh build-dist
```

It will: 
- compile jars containing Scala-based components 
  (currently only LAMLStringIndexer required for LE-family transformers)
  
- download Spark distro and use dockerfiles from there to build base pyspark images
  (and push these images to the remote docker repo)
  
- compile lama wheel (including spark subpackage) and build a docker image based upon mentioned above pyspark images
  (this image will be pushed to the remote repository too)
  
#### 5. Run an example on the remote cluster
To do that use the following command:
```shell
./bin/slamactl.sh submit-job ./examples/spark/tabular-preset-automl.py
```
The command submits a driver pod (using spark-submit) to the cluster which creates executor pods.

#### 6. Forward 4040 port to make Spark Web UI accessible.
The utility provides a command to make port forwording for the running example.
```shell
./bin/slamactl.sh port-forward ./examples/spark/tabular-preset-automl.py
```
The driver's 4040 port will be forwarded to http://localhost:9040.

#### Run on local Hadoop YARN
Copy lama wheel file from 'dist/LightAutoML-0.3.0-py3-none-any.whl' to 'docker-hadoop/nodemanager/LightAutoML-0.3.0-py3-none-any.whl'
```
cp dist/LightAutoML-0.3.0-py3-none-any.whl docker-hadoop/nodemanager/LightAutoML-0.3.0-py3-none-any.whl
```
Go to 'docker-hadoop' and configure docker-compose.yml. Add setting to mount directory with datasets to nodemanager1 service.
```
cd docker-hadoop

# see line with '- /opt/spark_data:/opt/spark_data' as example
```
Build image for 'nodemanager' service and 'spark-submit'.
```
make build-nodemanager-with-python
make build-image-to-spark-submit
```
Start Hadoop YARN services
```
docker-compose up
```
Send job to cluster via `spark-submit` container
```
docker exec -ti spark-submit bash -c "./bin/slamactl.sh submit-job-yarn examples/spark/tabular-preset-automl.py"
```
