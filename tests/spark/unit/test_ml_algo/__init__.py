import pytest
from pyspark.sql import SparkSession


# NOTE!!!
# All tests require PYSPARK_PYTHON env variable to be set
# for example: PYSPARK_PYTHON=/home/nikolay/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()
