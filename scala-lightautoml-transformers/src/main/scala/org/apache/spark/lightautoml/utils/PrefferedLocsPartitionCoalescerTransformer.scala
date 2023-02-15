package org.apache.spark.lightautoml.utils

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.StructType

class PrefferedLocsPartitionCoalescerTransformer(override val uid: String, val prefLoc: String) extends Transformer  {
  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.active
    val ds = dataset.asInstanceOf[Dataset[Row]]

    // real numPartitions is identified from the incoming dataset
    val coalesced_rdd = ds.rdd.coalesce(
      numPartitions = 100,
      shuffle = false,
      partitionCoalescer = Some(new PrefferedLocsPartitionCoalescer(prefLoc))
    )

    val coalesced_df = spark.createDataFrame(coalesced_rdd, schema = dataset.schema)

    coalesced_df
  }

  override def copy(extra: ParamMap): Transformer = new PrefferedLocsPartitionCoalescerTransformer(uid, prefLoc)

  override def transformSchema(schema: StructType): StructType = schema.copy()
}
