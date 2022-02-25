from pyspark.sql import SparkSession

# from pyspark.ml.feature import StringIndexer, StringIndexerModel
from lightautoml.spark.transformers.scala_wrappers.laml_string_indexer import LAMLStringIndexer, LAMLStringIndexerModel


JAR_PATH = "D:\\Projects\\Sber\\LAMA\\Sber-LAMA\\lightautoml\\spark\\transformers\\spark-lightautoml\\target\\scala-2.12\\spark-lightautoml_2.12-0.1.jar"


def test_laml_string_indexer():
    spark = (
        SparkSession
        .builder
        .appName("test_laml_string_indexer")
        .master("local[1]")
        .config("spark.jars", JAR_PATH)
        .getOrCreate()
    )

    file = "file:///D:\\Projects\\Sber\\LAMA\\Sber-LAMA-Stuff\\stringindexer-data\\data.json"
    test_file = "file:///D:\\Projects\\Sber\\LAMA\\Sber-LAMA-Stuff\\stringindexer-data\\test_data.json"

    df = spark.read.json(file).cache()
    test_df = spark.read.json(test_file).cache()

    indexer: LAMLStringIndexer = LAMLStringIndexer(
        inputCol="value",
        outputCol="index",
        handleInvalid="keep",
        minFreq=5,
        defaultValue=float("nan")
    )

    model: LAMLStringIndexerModel = indexer.fit(df)

    indexed_df = model.transform(test_df)

    print("-- Source DF --")
    df.show(100)

    print("-- Test DF --")
    test_df.show(100)

    print("-- Indexed DF --")
    indexed_df.show(100)

