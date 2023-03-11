package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, col, udf}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.ml.functions.vector_to_array

class TestAveragingTransformer extends AnyFunSuite with BeforeAndAfterAll with Logging {
  val num_workers = 3
  val num_cores = 2
  val folds_count = 5

  val spark: SparkSession = SparkSession
          .builder()
          .master(s"local[$num_cores, 1024]")
          .getOrCreate()

  import spark.implicits.StringToColumn

  override protected def afterAll(): Unit = {
    spark.stop()
  }

  private val vectorUdf = udf { vecs: Seq[DenseVector] =>
    val not_null_vecs = vecs.count(_ != null)
    val result = vecs.map { vec: Any =>
      vec match {
        case v if v == null => Vectors.dense(0.0, 0.0).asBreeze
        case v: Vector => v.asBreeze
        case v => throw new IllegalArgumentException(
          "function vector_to_array requires a non-null input argument and input type must be " +
                  "`org.apache.spark.ml.linalg.Vector` or `org.apache.spark.mllib.linalg.Vector`, " +
                  s"but got ${if (v == null) "null" else v.getClass.getName}.")
      }
    }.reduce(_ + _)
    val v = result.mapActiveValues(_ / not_null_vecs)
    new DenseVector(v.toArray)
  }

  test("AveragingTransformer") {
    val df = spark.read.parquet("/opt/tmp/not_averaged_dataset.parquet")

    val cols = (0 until 5).map(i => col(s"Mod_0_LightGBM_prediction_$i"))

    val new_df = df.select(vectorUdf(array(cols:_*)).alias("avg_predicts"))
    new_df.write.mode("overwrite").format("noop").save()
    val k = 0
  }
}
