package org.apache.spark.lightautoml.utils

import breeze.linalg
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{array, col, udf}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.types.{DataTypes, StructField, StructType, ArrayType, DoubleType}
import scala.collection.JavaConverters._

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

  private val vectorUdf = udf { (vecs: Seq[DenseVector], vec_size: Int) =>
    val not_null_vecs = vecs.count(_ != null)
    if (not_null_vecs == 0){
      null
    }
    else {
      val result = vecs.map { vec: Any =>
        vec match {
          case v if v == null => linalg.DenseVector.fill(vec_size)(0.0)
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
  }

  test("AveragingTransformer") {
    val numPreds = 5
    val dimSize = 3
    val fields = StructField(s"correct_answer", ArrayType(DataTypes.DoubleType)) :: (0 until numPreds)
            .map(i => StructField(s"pred_$i", ArrayType(DataTypes.DoubleType))).toList
    val schema = StructType(fields)
    val data = List(
      Row(Array(0.38, 0.23, 0.39), Array(0.38, 0.23, 0.39), null, null, null, null),
      Row(Array(0.38, 0.23, 0.39), null, null, Array(0.38, 0.23, 0.39), null, null),
      Row(null, null, null, null, null, null),
      Row(Array(0.445, 0.255, 0.3), Array(0.38, 0.23, 0.39), Array(0.21, 0.48, 0.31), Array(0.99, 0.01, 0.0), Array(0.2, 0.3, 0.5), null),
      Row(Array(0.4022, 0.3386, 0.2592),Array(0.38, 0.23, 0.39), Array(0.21, 0.48, 0.31), Array(0.99, 0.01, 0.0), Array(0.2, 0.3, 0.5), Array(0.231, 0.673, 0.096)),
    ).asJava

    val df = spark.createDataFrame(data, schema)

//    val df = spark.read.parquet("/opt/tmp/not_averaged_dataset.parquet")

    val cols = (0 until numPreds).map(i => col(s"Mod_0_LightGBM_prediction_$i"))

    val new_df = df.select($"correct_answer", vectorUdf(array(cols:_*)).alias("avg_predict"))
    df.write.mode("overwrite").format("noop").save()
    val k = 0
  }
}
