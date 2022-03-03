#!/usr/bin/env bash

set -e

APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

KUBE_NAMESPACE=spark-lama-exps

remote_script_path=run.py
scp examples/spark/tabular_preset_automl_copy.py \
  node2.bdcl:/mnt/ess_storage/DN_1/tmp/scripts-shared-vol/${remote_script_path}

ssh node2.bdcl "sudo chmod 755 /mnt/ess_storage/DN_1/tmp/scripts-shared-vol/${remote_script_path}"

spark-submit \
  --master k8s://${APISERVER} \
  --deploy-mode cluster \
  --conf 'spark.kubernetes.container.image=node2.bdcl:5000/spark-py-lama:lama-v3.2.0' \
  --conf 'spark.kubernetes.namespace='${KUBE_NAMESPACE} \
  --conf 'spark.kubernetes.authenticate.driver.serviceAccountName=spark' \
  --conf 'spark.kubernetes.memoryOverheadFactor=0.4' \
  --conf 'spark.kubernetes.driver.label.appname=driver-test-submit-run' \
  --conf 'spark.kubernetes.executor.label.appname=executor-test-submit-run' \
  --conf 'spark.kubernetes.executor.deleteOnTermination=false' \
  --conf 'spark.kubernetes.container.image.pullPolicy=Always' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --conf 'spark.jars=/home/nikolay/.ivy2/jars/com.azure_azure-ai-textanalytics-5.1.4.jar,/home/nikolay/.ivy2/jars/com.azure_azure-core-1.22.0.jar,/home/nikolay/.ivy2/jars/com.azure_azure-core-http-netty-1.11.2.jar,/home/nikolay/.ivy2/jars/com.azure_azure-storage-blob-12.14.2.jar,/home/nikolay/.ivy2/jars/com.azure_azure-storage-common-12.14.1.jar,/home/nikolay/.ivy2/jars/com.azure_azure-storage-internal-avro-12.1.2.jar,/home/nikolay/.ivy2/jars/com.beust_jcommander-1.27.jar,/home/nikolay/.ivy2/jars/com.chuusai_shapeless_2.12-2.3.2.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.core_jackson-annotations-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.core_jackson-core-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.core_jackson-databind-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.dataformat_jackson-dataformat-xml-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.datatype_jackson-datatype-jsr310-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.jackson.module_jackson-module-jaxb-annotations-2.12.5.jar,/home/nikolay/.ivy2/jars/com.fasterxml.woodstox_woodstox-core-6.2.4.jar,/home/nikolay/.ivy2/jars/com.github.vowpalwabbit_vw-jni-8.9.1.jar,/home/nikolay/.ivy2/jars/com.jcraft_jsch-0.1.54.jar,/home/nikolay/.ivy2/jars/com.linkedin.isolation-forest_isolation-forest_3.2.0_2.12-2.0.8.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-cognitive_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-core_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-deep-learning_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-lightgbm_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-opencv_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.azure_synapseml-vw_2.12-0.9.5.jar,/home/nikolay/.ivy2/jars/com.microsoft.cntk_cntk-2.4.jar,/home/nikolay/.ivy2/jars/com.microsoft.cognitiveservices.speech_client-jar-sdk-1.14.0.jar,/home/nikolay/.ivy2/jars/com.microsoft.ml.lightgbm_lightgbmlib-3.2.110.jar,/home/nikolay/.ivy2/jars/com.microsoft.onnxruntime_onnxruntime_gpu-1.8.1.jar,/home/nikolay/.ivy2/jars/commons-codec_commons-codec-1.10.jar,/home/nikolay/.ivy2/jars/commons-logging_commons-logging-1.2.jar,/home/nikolay/.ivy2/jars/io.netty_netty-buffer-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-codec-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-codec-dns-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-codec-http2-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-codec-http-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-codec-socks-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-common-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-handler-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-handler-proxy-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-resolver-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-resolver-dns-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-resolver-dns-native-macos-4.1.68.Final-osx-x86_64.jar,/home/nikolay/.ivy2/jars/io.netty_netty-tcnative-boringssl-static-2.0.43.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-transport-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.netty_netty-transport-native-epoll-4.1.68.Final-linux-x86_64.jar,/home/nikolay/.ivy2/jars/io.netty_netty-transport-native-kqueue-4.1.68.Final-osx-x86_64.jar,/home/nikolay/.ivy2/jars/io.netty_netty-transport-native-unix-common-4.1.68.Final.jar,/home/nikolay/.ivy2/jars/io.projectreactor.netty_reactor-netty-core-1.0.11.jar,/home/nikolay/.ivy2/jars/io.projectreactor.netty_reactor-netty-http-1.0.11.jar,/home/nikolay/.ivy2/jars/io.projectreactor_reactor-core-3.4.10.jar,/home/nikolay/.ivy2/jars/io.spray_spray-json_2.12-1.3.2.jar,/home/nikolay/.ivy2/jars/jakarta.activation_jakarta.activation-api-1.2.1.jar,/home/nikolay/.ivy2/jars/jakarta.xml.bind_jakarta.xml.bind-api-2.3.2.jar,/home/nikolay/.ivy2/jars/org.apache.httpcomponents_httpclient-4.5.6.jar,/home/nikolay/.ivy2/jars/org.apache.httpcomponents_httpcore-4.4.10.jar,/home/nikolay/.ivy2/jars/org.apache.httpcomponents_httpmime-4.5.6.jar,/home/nikolay/.ivy2/jars/org.apache.spark_spark-avro_2.12-3.2.0.jar,/home/nikolay/.ivy2/jars/org.beanshell_bsh-2.0b4.jar,/home/nikolay/.ivy2/jars/org.codehaus.woodstox_stax2-api-4.2.1.jar,/home/nikolay/.ivy2/jars/org.openpnp_opencv-3.2.0-1.jar,/home/nikolay/.ivy2/jars/org.reactivestreams_reactive-streams-1.0.3.jar,/home/nikolay/.ivy2/jars/org.scalactic_scalactic_2.12-3.0.5.jar,/home/nikolay/.ivy2/jars/org.scala-lang_scala-reflect-2.12.4.jar,/home/nikolay/.ivy2/jars/org.slf4j_slf4j-api-1.7.32.jar,/home/nikolay/.ivy2/jars/org.spark-project.spark_unused-1.0.0.jar,/home/nikolay/.ivy2/jars/org.testng_testng-6.8.8.jar,/home/nikolay/.ivy2/jars/org.tukaani_xz-1.8.jar,/home/nikolay/.ivy2/jars/org.typelevel_macro-compat_2.12-1.1.1.jar,jars/spark-lightautoml_2.12-0.1.jar' \
  --conf 'spark.driver.cores=4' \
  --conf 'spark.driver.memory=32g' \
  --conf 'spark.executor.instances=4' \
  --conf 'spark.executor.cores=4' \
  --conf 'spark.executor.memory=16g' \
  --conf 'spark.cores.max=16' \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.kubernetes.file.upload.path=/mnt/nfs/spark_upload_dir' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.options.claimName=scripts-shared-vol' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.mount.path=/scripts/' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts-shared-vol.mount.readOnly=true' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.claimName=exp-results-vol' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.path=/exp_results' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.exp-results-vol.mount.readOnly=false' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.claimName=mnt-nfs' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass=nfs' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.path=/mnt/nfs/' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly=false' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.claimName=mnt-nfs' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass=nfs' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.path=/mnt/nfs/' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly=false' \
  local:///scripts/${remote_script_path}

# --conf 'spark.jars=jars/spark-lightautoml_2.12-0.1.jar' \
#  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
#  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \