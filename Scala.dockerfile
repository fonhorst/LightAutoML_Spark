FROM ubuntu:20.04

# Install required utils & Amazon Java 8 JDK
RUN apt-get update && \
    apt-get install -y wget apt-transport-https curl gnupg software-properties-common && \
    wget -O- https://apt.corretto.aws/corretto.key | apt-key add - && \
    add-apt-repository 'deb https://apt.corretto.aws stable main' && \
    apt-get update && \
    apt-get install -y java-1.8.0-amazon-corretto-jdk

# Install SBT
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" \
    | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import && \
    chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg && \
    apt-get update && \
    apt-get install -y sbt

RUN rm -rf /var/lib/apt/lists/*

# Copy source code. Instead of copying every time you are able to mount source code directory in this path.
COPY lightautoml/spark/transformers/spark-lightautoml /lightautoml/spark/transformers/spark-lightautoml

# Just for cache the spark packages
RUN cd /lightautoml/spark/transformers/spark-lightautoml && \
    sbt clean && \
    sbt package && \
    sbt clean

# Command on container startup: build a JAR file
# JAR file will be located at: /lightautoml/spark/transformers/spark-lightautoml/target/scala-2.12/spark-lightautoml_2.12-{VERSION}.jar
CMD cd /lightautoml/spark/transformers/spark-lightautoml && \
    sbt clean && \
    sbt package


