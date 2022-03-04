FROM ubuntu:20.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
#ARG TENSORFLOW_VERSION=2.5.0
ARG PYTORCH_VERSION=1.8.1
ARG PYTORCH_LIGHTNING_VERSION=1.2.9
ARG TORCHVISION_VERSION=0.9.1
#ARG MXNET_VERSION=1.8.0.post0

ARG PYSPARK_PACKAGE=pyspark==3.2.0
ARG SPARK_PACKAGE=spark-3.2.0/spark-3.2.0-bin-hadoop2.7.tgz

ARG PYTHON_VERSION=3.9

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk-headless \
        openssh-client \
        openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
    cd /usr/local && \
    tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && \
    ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch, TensorFlow, Keras and MXNet
RUN pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}
RUN pip install --no-cache-dir pytorch_lightning==${PYTORCH_LIGHTNING_VERSION}

RUN pip install --no-cache-dir future typing packaging
#RUN pip install --no-cache-dir \
#    tensorflow-cpu==${TENSORFLOW_VERSION} \
#    keras \
#    h5py

#RUN pip install --no-cache-dir mxnet==${MXNET_VERSION}

# Install Spark stand-alone cluster.
RUN wget --progress=dot:giga "https://www.apache.org/dyn/closer.lua/spark/${SPARK_PACKAGE}?action=download" -O - | tar -xzC /tmp; \
    archive=$(basename "${SPARK_PACKAGE}") bash -c "mv -v /tmp/\${archive/%.tgz/} /spark"

# Install PySpark.
RUN pip install --no-cache-dir ${PYSPARK_PACKAGE}

RUN pip install --no-cache-dir horovod==0.22.0
