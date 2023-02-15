package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}
import org.apache.spark.sql.SparkSession

class PrefferedLocsPartitionCoalescer(val prefLoc: String) extends PartitionCoalescer with Serializable{
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val spark = SparkSession.active
    val cores = 2//spark.conf.get("spark.executor.cores").toInt

    val gr_size = (parent.partitions.length / cores).ceil.toInt

    val result = parent.partitions.grouped(gr_size).map{ ps =>
      val pg = new PartitionGroup(Some(prefLoc))
      ps.foreach(p => pg.partitions += p)
      pg
    }.toArray

    result
//    parent.partitions.map{ p =>
//      val pg = new PartitionGroup(Some(prefLoc))
//      pg.partitions += p
//      pg
//    }.toArray
  }
}
