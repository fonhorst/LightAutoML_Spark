import random
import time

from pyspark.sql import SparkSession
from pyspark.sql.pandas.functions import pandas_udf

import pandas as pd

spark = (
    SparkSession
        .builder
        .master('local[4]')
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
)

data = [
    {"a": i, "b": i * 10, "c": i * 100}
    for i in range(100)
]

df = spark.createDataFrame(data)
df = df.cache()
df.write.mode('overwrite').format('noop').save()

mapping_size = 1_000_000
bdata = {i: random.randint(0, 1000) for i in range(mapping_size)}
bval = spark.sparkContext.broadcast(bdata)


@pandas_udf('int')
def func1(col: pd.Series) -> pd.Series:
    mapping = bval.value
    msize = len(mapping)

    return col.apply(lambda x: x + mapping[x] if x in mapping else 0.0)


df_1 = df.select([func1(c).alias(c) for c in df.columns])
df_1 = df_1.cache()
df_1.write.mode('overwrite').format('noop').save()


@pandas_udf('int')
def func2(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: x - 10)


df_2 = df_1.select([func2(c).alias(c) for c in df_1.columns])
df_2 = df_2.cache()
df_2.write.mode('overwrite').format('noop').save()


print("Finished")
time.sleep(600)

spark.stop()