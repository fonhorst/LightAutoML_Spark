package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}
import org.apache.spark.sql.SparkSession

class PrefferedLocsPartitionCoalescer(val prefLoc: String) extends PartitionCoalescer with Serializable{
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val spark = SparkSession.active
    val cores = spark.conf.get("spark.executor.cores").toInt

    parent.partitions.sliding(cores).map{ ps =>
      val pg = new PartitionGroup(Some(prefLoc))
      ps.foreach(p => pg.partitions += p)
      pg
    }.toArray
  }
}
