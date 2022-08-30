from importlib_metadata import version
from packaging.version import parse

if parse(version('pyspark')) >= parse('3.1.0'):
    from pyspark.ml.functions import array_to_vector
else:
    from pyspark import SparkContext
    from pyspark.sql import Column
    from pyspark.sql.column import _to_java_column

    def array_to_vector(col):
        sc = SparkContext._active_spark_context
        return Column(
            sc._jvm.org.apache.spark.lightautoml.utils.functions.array_to_vector(_to_java_column(col)))

array_to_vector = array_to_vector
