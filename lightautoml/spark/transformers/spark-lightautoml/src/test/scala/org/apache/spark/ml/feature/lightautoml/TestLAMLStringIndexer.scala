package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.lightautoml.LAMLStringIndexer


object TestLAMLStringIndexer extends App {

  val file = "file:///D:\\Projects\\Sber\\LAMA\\Sber-LAMA-Stuff\\stringindexer-data\\data.json"
  val testFile = "file:///D:\\Projects\\Sber\\LAMA\\Sber-LAMA-Stuff\\stringindexer-data\\test_data.json"
  val spark = SparkSession
          .builder()
          .appName("test")
          .master("local[1]")
          .getOrCreate()

  //import spark.sqlContext.implicits._

  val df = spark.read.json(file).cache()
  val testDf = spark.read.json(testFile).cache()
  val cnt = df.count()

  println("-- Source --")
  df.show(100)

  val startTime = System.currentTimeMillis()

  val indexer = new StringIndexer().setInputCol("value").setOutputCol("index").setHandleInvalid("keep")
  println(indexer.uid)

  val model = indexer.fit(df)
  val testIndexed = model.transform(testDf)

  println("-- Spark Indexed --")
  testIndexed.show(100)

  val lamaIndexer = new LAMLStringIndexer()
          .setMinFreq(Array(5))
          .setDefaultValue(-1.0F)
          .setInputCols(Array("value"))
          .setOutputCols(Array("index"))
          .setHandleInvalid("keep")

  val _lamaModelTestNoRuntimeError = new LAMLStringIndexerModel(labelsArray = Array(Array("a", "b")))

  val _sparkModelTestNoRuntimeError = new StringIndexerModel(labelsArray = Array(Array("a", "b")))

  println(lamaIndexer.uid)

  val lamaModel = lamaIndexer.fit(df)
  val lamaTestIndexed = lamaModel.transform(testDf)

  println("-- Lama Indexed --")
  lamaTestIndexed.show(100)

  val endTime = System.currentTimeMillis()

  println(s"Duration = ${(endTime - startTime) / 1000D} seconds")
  println(s"Size: ${cnt}")

  println(s"[${indexer.uid} - ${model.uid}] // [${lamaIndexer.uid} - ${lamaModel.uid}]")

  while (args(0).toBoolean) {
    Thread.sleep(1000)
  }


}
