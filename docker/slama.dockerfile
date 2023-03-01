FROM debian:bookworm-20230227

ARG PYTHON_VERSION=3.9.16

# installing common dependencies
RUN apt-get update && apt-get install -y coreutils build-essential gcc g++ gdb lcov pkg-config libbz2-dev \
    libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
    wget curl nano net-tools libsnappy-dev

# installing python
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzvf Python-${PYTHON_VERSION}.tgz

RUN cd Python-${PYTHON_VERSION} && ./configure --prefix=/usr/local && make && make install

# installing java
RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz

RUN tar -xvf openjdk-11+28_linux-x64_bin.tar.gz

RUN mv jdk-11 /usr/local/lib/jdk-11

RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java

# installing slama requirements
COPY ./requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ARG SLAMA_BUILD_DIR=.

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate();'

# installing slama
COPY $SLAMA_BUILD_DIR/examples/spark /src/examples-spark

RUN mkdir /src/examples-spark/jars

COPY $SLAMA_BUILD_DIR/jars/spark-lightautoml_2.12-0.1.jar /src/examples-spark/jars/

COPY $SLAMA_BUILD_DIR/dist/SparkLightAutoML_DEV-0.3.0-py3-none-any.whl /src/SparkLightAutoML_DEV-0.3.0-py3-none-any.whl

RUN pip3 install /src/SparkLightAutoML_DEV-0.3.0-py3-none-any.whl

# set execution settings
WORKDIR /src/examples-spark

ENTRYPOINT ["python3"]
