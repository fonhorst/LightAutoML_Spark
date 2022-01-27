FROM spark-py:pyspark-executor-3.2.0

ARG spark_jars_cache=jars_cache

USER root

RUN mkdir -p /src

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt

COPY ${spark_jars_cache} /root/.ivy2/cache

COPY dist/LightAutoML-0.3.0-py3-none-any.whl /tmp/LightAutoML-0.3.0-py3-none-any.whl
RUN pip install /tmp/LightAutoML-0.3.0-py3-none-any.whl

USER ${spark_id}
