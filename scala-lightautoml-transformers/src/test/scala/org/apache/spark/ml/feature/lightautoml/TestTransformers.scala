package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.internal.Logging
import org.apache.spark.internal.config.Tests.IS_TESTING
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.lightautoml.{LAMLStringIndexer, LAMLStringIndexerModel}
import org.apache.spark.util.AccumulatorContext
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach}
import org.scalatest.funsuite.AnyFunSuite

import java.util.{Locale, TimeZone}


abstract class BaseFunSuite extends AnyFunSuite with BeforeAndAfterAll with Logging {
  // scalastyle:on

  // Initialize the logger forcibly to let the logger log timestamp
  // based on the local time zone depending on environments.
  // The default time zone will be set to America/Los_Angeles later
  // so this initialization is necessary here.
  log

  var spark: SparkSession = _

  protected override def beforeAll(): Unit = {
    spark =
      SparkSession
              .builder()
              .appName("test")
              .master("local[1]")
              .getOrCreate()
  }

  protected override def afterAll(): Unit = {
    spark.stop()
  }
}

class TestLAMLStringIndexer extends BaseFunSuite {

  test("Smoke LAMLStringIndexer test") {
    val file = "resources/data.json"
    val testFile = "resources/test_data.json"


    val df = spark.read.json(file).cache()
    val testDf = spark.read.json(testFile).cache()
    val cnt = df.count()

    println("-- Source --")
    df.show(100)

    val startTime = System.currentTimeMillis()

    val lamaIndexer = new LAMLStringIndexer()
            .setMinFreq(Array(1))
            .setFreqLabel(true)
            .setDefaultValue(1.0F)
            .setInputCols(Array("value"))
            .setOutputCols(Array("index"))
            .setHandleInvalid("keep")

    println(lamaIndexer.uid)

    val lamaModel = lamaIndexer.fit(df)
    val lamaTestIndexed = lamaModel.transform(testDf)

    println("-- Lama Indexed --")
    lamaTestIndexed.show(100)

    val endTime = System.currentTimeMillis()

    println(s"Duration = ${(endTime - startTime) / 1000D} seconds")
    println(s"Size: ${cnt}")

    lamaModel.write.overwrite().save("/tmp/LAMLStringIndexerModel")
    val pipelineModel = LAMLStringIndexerModel.load("/tmp/LAMLStringIndexerModel")
    pipelineModel.transform(testDf)
  }

  test("Smoke test of Target Encoder transformer") {

  }
}
