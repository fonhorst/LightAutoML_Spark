package org.apache.spark.lightautoml.utils

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.util.Random

object TestBalancedUnionPartitionCoalescer extends App {
  val num_workers = 3
  val num_cores = 2
  val folds_count = 5

  val spark = SparkSession
          .builder()
          .master(s"local-cluster[${num_workers}, ${num_cores}, 1024]")
          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.jar")
          .getOrCreate()

  import spark.sqlContext.implicits._

  val df = spark
          .sparkContext.parallelize((0 until 5000)
          .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
          .repartition(num_workers * num_cores * 2)
          .cache()
  df.write.mode("overwrite").format("noop").save()

  val dfs = (0 until 5).map(x => df.where(col("fold").equalTo(x)))
  val full_df = dfs.reduce((acc, sdf) => acc.unionByName(sdf))

  val coalesced_rdd = full_df.rdd.coalesce(
    df.rdd.getNumPartitions,
    shuffle = false,
    partitionCoalescer = Some(new BalancedUnionPartitionCoalescer)
  )

  //  coalesced_rdd.count()

  var coalesced_df = spark.createDataFrame(coalesced_rdd, schema = full_df.schema)

  coalesced_df = coalesced_df.cache()
  coalesced_df.write.mode("overwrite").format("noop").save()

  coalesced_df.count()

  val result = coalesced_df.rdd.collectPartitions()

  // check for balanced dataset:
  // 1. all executors should have the same number partitions as their parents dataset have
  // 2. all partitions should have approximately the same number of records

  val sameNumOfPartitions = df.rdd.getNumPartitions == coalesced_df.rdd.getNumPartitions
  assert(sameNumOfPartitions)

  val parts_sizes = result.map(_.length)
  val min_size = parts_sizes.min
  val max_size = parts_sizes.max
  assert((max_size - min_size).toFloat / min_size <= 0.02)

  // part_id, (fold, record_count)
  val partsWithFolds = result
          .zipWithIndex
          .flatMap(x => x._1.map(y => (x._2, y.getInt(1))))
          .sortBy(_._1)
          .groupBy(_._1)
          .map(x => (x._1, x._2.map(_._2)))
          .map(x => (x._1, x._2.sortBy(x => x).groupBy(x => x).map(x => (x._1, x._2.length))))

  val allFoldsInAllPartitions = partsWithFolds.forall(_._2.size == folds_count)
  assert(allFoldsInAllPartitions)

  val foldsBalancedInAllPartitions = partsWithFolds.forall{ x =>
    val min_count = x._2.values.min
    val max_count = x._2.values.max
    (max_count - min_count).toFloat / min_count <= 0.02
  }
  assert(foldsBalancedInAllPartitions)

  spark.stop()
}
