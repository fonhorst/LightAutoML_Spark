import os
import uuid
from typing import cast

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator

spark = (
    SparkSession
    .builder
    .master("local-cluster[2, 2, 4096]")
    .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
    .config("spark.kryoserializer.buffer.max", "512m")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

data_file = "/opt/spark_data/test-le.parquet"
if not os.path.exists(data_file):
    print("Data file does not exist. Creating new one.")
    uuids = [{"uid": str(uuid.uuid4()) * 25, "uid_2": str(uuid.uuid4()) * 25} for _ in range(100_000)]
    df = spark.createDataFrame(uuids)
    df.write.mode('overwrite').parquet(data_file)

df = spark.read.parquet(data_file)
df = df.cache()
df.write.mode('overwrite').format('noop').save()

estimator = SparkLabelEncoderEstimator(
    input_cols=['uid', 'uid_2'],
    input_roles={'uid': CategoryRole(unknown=0), 'uid_2': CategoryRole(unknown=0)}
)

print("Fitting LE estimator")
transformer = estimator.fit(df)
print("Finished LE estimator")

t_df = cast(DataFrame, transformer.transform(df))
# t_df = t_df.localCheckpoint(eager=True)
t_df = t_df.cache()
t_df.write.mode('overwrite').format('noop').save()
print("Finished LE transform")

t_df.select([F.col(c) * 2 for c in t_df.columns]).write.mode('overwrite').format('noop').save()


t_df_2 = t_df.localCheckpoint(eager=True)
print("Checkpointing finished")
t_df_2.select([F.col(c) * 2 for c in t_df.columns]).write.mode('overwrite').format('noop').save()
# import time
# time.sleep(600)

spark.stop()