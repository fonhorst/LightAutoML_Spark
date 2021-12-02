from pyspark.sql import SparkSession

from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.tasks import Task
from . import spark


def test_spark_reader(spark: SparkSession):

    df = spark.read.csv("../../../examples/data/sampled_app_train.csv", header=True)

    sreader = SparkToSparkReader(task=Task("binary"))

    sdf = sreader.fit_read(df)