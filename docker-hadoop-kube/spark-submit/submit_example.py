import time

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

result = spark.sparkContext.parallelize([i for i in range(10)]).sum()
print(f"Test result: {result}")

time.sleep(600)

spark.stop()
