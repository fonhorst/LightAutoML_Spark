import time
from datetime import datetime

from decorator import contextmanager
from pyspark.sql import SparkSession


@contextmanager
def spark_session(parallelism: int = 1) -> SparkSession:
    spark = (
        SparkSession
        .builder
        .master(f"local[{parallelism}]")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    try:
        yield spark
    finally:
        # time.sleep(600)
        spark.stop()


@contextmanager
def print_exec_time():
    start = datetime.now()
    yield
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"Exec time: {duration}")