package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD, UnionPartition}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.util.Random


class BalancedUnionPartitionCoalescer extends PartitionCoalescer with Serializable {
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val up_arr = parent.partitions.map(_.asInstanceOf[UnionPartition[_]])
    val parent2parts = up_arr
            .groupBy(_.parentRddIndex)
            .map{case(parentRddIndex, ups) => (parentRddIndex, ups.sortBy(_.parentPartition.index))}

    val parent2size = parent2parts.map{case(parentRddIndex, ups) => (parentRddIndex, ups.length)}
    val unique_sizes = parent2size.values.toSet

    assert(
      unique_sizes.size == 1,
      s"Found differences in num of parts: $unique_sizes. Parent to parts num mapping: $parent2size"
    )

    val partsNum = unique_sizes.head

//    assert(maxPartitions <= partsNum)

    val pgs = (0 until partsNum).map(i => {
      val pg = new PartitionGroup()
      parent2parts.values.foreach(x => pg.partitions += x(i))
      pg
    })

    pgs.toArray
  }
}

