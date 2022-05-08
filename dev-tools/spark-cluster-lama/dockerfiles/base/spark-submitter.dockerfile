FROM node2.bdcl:5000/spark-base:3.2.1-py3.9

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

COPY build_tmp/dist/LightAutoML-0.3.0.tar.gz /opt
COPY build_tmp/spark-lightautoml_2.12-0.1.jar /opt

RUN pip install /opt/LightAutoML-0.3.0.tar.gz

COPY submit.sh /opt
COPY build_tmp/examples /examples

RUN chmod 777 /opt/bitnami/spark
WORKDIR /opt/bitnami/spark
USER 1001
ENTRYPOINT [ "/opt/submit.sh" ]
CMD [ "/examples/spark/tabular-preset-automl.py" ]
