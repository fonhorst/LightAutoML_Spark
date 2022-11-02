package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import java.io.File
import scala.collection.JavaConverters._
import scala.reflect.io.Directory


abstract class BaseFunSuite extends AnyFunSuite with BeforeAndAfterAll with Logging {
  // scalastyle:on

  // Initialize the logger forcibly to let the logger log timestamp
  // based on the local time zone depending on environments.
  // The default time zone will be set to America/Los_Angeles later
  // so this initialization is necessary here.
  log

  var spark: SparkSession = _

  val workdir: String = "/tmp/test_transformers"

  protected override def beforeAll(): Unit = {
    spark =
      SparkSession
              .builder()
              .appName("test")
              .master("local[1]")
              .getOrCreate()


    val dir = new Directory(new File(workdir))
    dir.createDirectory()
  }

  protected override def afterAll(): Unit = {
    spark.stop()

    val dir = new Directory(new File(workdir))
    dir.deleteRecursively()
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

    val enc = Map(
      ("a", Array(0.0, -1.0, -2.0, -3.0, -4.0)),
      ("b", Array(0.0, -1.0, -2.0)),
      ("c", Array(0.0, -1.0, -2.0, -3.0))
    )

    val oof_enc = Map(
      ("a", Array(
        Array(0.0, 10.0, 20.0, 30.0, 40.0),
        Array(0.0, 11.0, 12.0, 13.0, 14.0),
        Array(0.0, 21.0, 22.0, 23.0, 24.0)
      )),
      ("b", Array(
        Array(0.0, 10.0, 20.0),
        Array(0.0, 11.0, 12.0),
        Array(0.0, 21.0, 22.0))
      ),
      ("c", Array(
        Array(0.0, 10.0, 20.0, 30.0),
        Array(0.0, 11.0, 12.0, 13.0),
        Array(0.0, 21.0, 22.0, 23.0)
      ))
    )

    val fold_column = "fold"
    val id_column = "id"

    // id, fold_column, some_other_col, a, b, c
    val data = Seq(
      Row(Seq(0, 0, 42, 1, 1, 1)),
      Row(Seq(1, 0, 43, 2, 1, 3)),
      Row(Seq(2, 1, 44, 1, 2, 3)),
      Row(Seq(3, 1, 45, 1, 2, 2)),
      Row(Seq(4, 2, 46, 3, 1, 1)),
      Row(Seq(5, 2, 47, 4, 1, 2)),
    ).toList.asJava

    val result_enc = Seq(
      Row(Seq(0, 0, 42, -1, -1, -1)),
      Row(Seq(1, 0, 43, -2, -1, -3)),
      Row(Seq(2, 1, 44, -1, -2, -3)),
      Row(Seq(3, 1, 45, -1, -2, -2)),
      Row(Seq(4, 2, 46, -3, -1, -1)),
      Row(Seq(5, 2, 47, -4, -1, -2)),
    )

    val result_oof_enc = Seq(
      Row(Seq(0, 0, 42, 10, 10, 10)),
      Row(Seq(1, 0, 43, 20, 10, 20)),
      Row(Seq(2, 1, 44, 11, 12, 13)),
      Row(Seq(3, 1, 45, 11, 12, 12)),
      Row(Seq(4, 2, 46, 23, 21, 21)),
      Row(Seq(5, 2, 47, 24, 21, 22)),
    )

    val schema = StructType(
      Array(
        StructField(id_column, IntegerType),
        StructField(fold_column, IntegerType),
        StructField("some_other_col", IntegerType)
      )
      ++ in_cols.map(col => StructField(col, IntegerType))
    )

    def checkResult(tdf: DataFrame, df: DataFrame, target_data: Seq[Row]): Unit = {
      tdf.columns should contain allElementsOf df.columns
      tdf.columns should contain allElementsOf out_cols
      out_cols.foreach(col => tdf.schema(col) shouldBe a [DoubleType])

      val resul_rows = tdf.orderBy(id_column).collect()
      resul_rows.zip(target_data).foreach {
        case (row, target) => row.toSeq should equal (target.toSeq)
      }
    }

    val df = spark.createDataFrame(data, schema)

    val te = new TargetEncoderTransformer("te_tr", enc, oof_enc, fold_column)
            .setInputCols(in_cols)
            .setOutputCols(out_cols)

    checkResult(te.transform(df), df, result_oof_enc)
    checkResult(te.transform(df), df, result_enc)
    checkResult(te.transform(df), df, result_enc)

    val path = workdir + "target_encoder.transformer"
    te.save(path)

    val loaded_te = TargetEncoderTransformer.load(path)
    checkResult(loaded_te.transform(df), df, result_enc)
    checkResult(loaded_te.transform(df), df, result_enc)
  }
}
