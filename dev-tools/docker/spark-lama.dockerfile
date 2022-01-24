FROM spark-pyspark-python:3.9-3.2.0

ARG spark_jars_cache=jars_cache

WORKDIR /src

RUN pip install poetry
RUN poetry config virtualenvs.create false --local

# we need star here to make copying of poetry.lock conditional
COPY requirements.txt /src

# workaround to make poetry not so painly slow on dependency resolution
# before this image building: poetry export -f requirements.txt > requirements.txt
RUN pip install -r requirements.txt

COPY ${spark_jars_cache} /root/.ivy2/cache

COPY dist/LightAutoML-0.3.0-py3-none-any.whl /tmp/LightAutoML-0.3.0-py3-none-any.whl
RUN pip install /tmp/LightAutoML-0.3.0-py3-none-any.whl


