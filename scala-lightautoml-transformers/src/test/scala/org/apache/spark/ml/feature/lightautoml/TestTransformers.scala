package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import scala.collection.JavaConverters._


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
    println(s"Size: $cnt")

    lamaModel.write.overwrite().save("/tmp/LAMLStringIndexerModel")
    val pipelineModel = LAMLStringIndexerModel.load("/tmp/LAMLStringIndexerModel")
    pipelineModel.transform(testDf)
  }

  test("Smoke test of Target Encoder transformer") {
    val in_cols = Seq("a", "b", "c").toArray
    val out_cols = in_cols.map(x => s"te_$x")

    val enc = in_cols
            .zipWithIndex.map{case (col, idx) => (col,(idx until idx + 4).map(_.toDouble).toArray)}
            .toMap

    val oof_enc = in_cols
            .zipWithIndex.map {
              case (col, idx) => (col, (1 until 2).map(
                i => (idx * i * 10 until idx * i * 10 + 4).map(_.toDouble).toArray).toArray
              )
            }
            .toMap

    val fold_column = "fold"

    // fold_column, some_other_col, a, b, c
    val data = (1 until 10).map(_ => Row(Seq(0, 42, 1, 1, 1))).toList.asJava
    val schema = StructType(
      Array(StructField(fold_column, IntegerType), StructField("some_other_col", IntegerType))
      ++ in_cols.map(col => StructField(col, IntegerType))
    )

    val df = spark.createDataFrame(data, schema)

    val te = new TargetEncoderTransformer("te_tr", enc, oof_enc, fold_column)
            .setInputCols(in_cols)
            .setOutputCols(out_cols)
    val tdf = te.transform(df)

    tdf.columns should contain allElementsOf df.columns
    tdf.columns should contain allElementsOf out_cols
    out_cols.foreach(col => tdf.schema(col) shouldBe a [DoubleType])

    val tdf_2 = te.transform(df)

    tdf_2.columns should contain allElementsOf df.columns
    tdf_2.columns should contain allElementsOf out_cols
    out_cols.foreach(col => tdf_2.schema(col) shouldBe a [DoubleType])
  }
}
