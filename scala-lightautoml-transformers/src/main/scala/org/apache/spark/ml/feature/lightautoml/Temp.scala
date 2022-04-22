package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD, UnionPartition}
import org.apache.spark.sql.SparkSession

import scala.util.Random
import org.apache.spark.sql.functions.col


class BalancedUnionPartitionCoalescer extends PartitionCoalescer with Serializable {
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val up_arr = parent.partitions.map(_.asInstanceOf[UnionPartition[_]])
    val parent2parts = up_arr
            .map(x => (x.parentRddIndex, x))
            .groupBy(_._1)
            .map(x => (x._1, x._2.map(_._2).sortBy(_.parentPartition.index)))

    val unique_sizes = parent2parts.map(_._2.length).toSet

    assert(unique_sizes.size == 1)

    val partsNum = unique_sizes.head

    assert(maxPartitions <= partsNum)

    val pgs = (0 until partsNum).map(i => {
      val pg = new PartitionGroup()
      parent2parts.values.foreach(x => pg.partitions += x(i))
      pg
    })

    pgs.toArray
  }
}


object Temp extends App {
  println("Hello")

  val spark = SparkSession.builder().master("local[2]").getOrCreate()

  import spark.sqlContext.implicits._

  val df = spark.sparkContext.parallelize((0 until 5000).map(x => (x, Random.nextInt(5)))).toDF("data", "fold")
  val dfs = (0 until 5).map(x => df.where(col("fold").equalTo(x)))
  val full_df = dfs.reduce((acc, sdf) => acc.unionByName(sdf))

  val coalesced_rdd = full_df.rdd.coalesce(
    df.rdd.getNumPartitions,
    shuffle = false,
    partitionCoalescer = Some(new BalancedUnionPartitionCoalescer)
  )

  val coalesced_df = spark.createDataFrame(coalesced_rdd, schema = full_df.schema)

  spark.stop()
}
