import os
import pickle
import sys
import uuid
from typing import cast, Iterator

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.pandas.functions import pandas_udf

from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator

import pandas as pd
import random

spark = (
    SparkSession
    .builder
    .master("local-cluster[2, 2, 8192]")
    .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
    .config("spark.kryoserializer.buffer.max", "512m")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

if not os.path.exists("/tmp/blob.bin"):
    with open("/tmp/blob.bin", "w") as f:
        arr = [i for i in range(10_000_000)]
        arr = pickle.dumps(arr)
        f.write(str(arr))
else:
    with open("/tmp/blob.bin", "r") as f:
        arr = f.read()

print("Generated data for Heavy broadcast")

data_file = "/opt/spark_data/test-le.parquet"
if not os.path.exists(data_file):
    print("Data file does not exist. Creating new one.")
    uuids = [{"uid": str(uuid.uuid4()) * 25, "uid_2": str(uuid.uuid4()) * 25} for _ in range(100_000)]
    df = spark.createDataFrame(uuids)
    df.write.mode('overwrite').parquet(data_file)

df = spark.read.parquet(data_file)

# df = df.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(3)])))
# df = df.drop("new_col")
# df = df.cache()
# df.write.mode('overwrite').format('noop').save()
# df.write.mode('overwrite').parquet(data_file)
# sys.exit(0)

df = df.cache()
df.write.mode('overwrite').format('noop').save()

print("Materialized the initial dataset")

# bcast_arr = spark.sparkContext.broadcast(arr)

@pandas_udf("string")
def func(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # data_len = len(bcast_arr.value)
    data_len = len(arr)
    for s in iterator:
        # yield s.str.len() + data_len
        # yield s.apply(lambda x: x + arr[random.randint(0, len(arr) - 1)])
        yield s

@pandas_udf("string")
def func2(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for s in iterator:
        # yield s.str.len() + data_len
        # yield s.apply(lambda x: x + arr[random.randint(0, len(arr) - 1)])
        yield s

t_df = df.select(func("uid").alias("uid_m"), func("uid_2").alias("uid_2_m"))
t_df.write.mode('overwrite').format('noop').save()

print("Make transforming (without caching)")

temp_df = df.select(func2("uid").alias("uid_m"), func2("uid_2").alias("uid_2_m"))
temp_df.write.mode('overwrite').format('noop').save()

print("Make simple transforming (without caching)")

t_df = t_df.cache()
t_df.write.mode('overwrite').format('noop').save()

df.unpersist(blocking=True)

print("Make transforming (with caching)")

# tt_df = t_df.select(F.length("uid_m").alias("uid_m"), F.length("uid_2_m").alias("uid_2_m"))
tt_df = t_df.select(func2("uid_m").alias("uid_m"), func2("uid_2_m").alias("uid_2_m"))
tt_df.write.mode('overwrite').format('noop').save()

print("Make secondary transforming")

tt_df = t_df.select(F.length("uid_m").alias("uid_m"), F.length("uid_2_m").alias("uid_2_m"))
tt_df.write.mode('overwrite').format('noop').save()

print("Make third transforming")


tt_df = t_df.localCheckpoint(eager=True)
print("Made localCheckpoint")

tt_df.select(F.length("uid_m").alias("uid_m"), F.length("uid_2_m").alias("uid_2_m"))
tt_df.write.mode('overwrite').format('noop').save()

import time
time.sleep(600)

spark.stop()