package org.apache.spark.lightautoml.utils

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.udf

object functions {
  private val arrayToVectorUdf = udf { array: Seq[Double] =>
    Vectors.dense(array.toArray)
  }

  /**
   * Converts a column of array of numeric type into a column of dense vectors in MLlib.
   * @param v: the column of array&lt;NumericType&gt type
   * @return a column of type `org.apache.spark.ml.linalg.Vector`
   * @since 3.1.0
   */
  def array_to_vector(v: Column): Column = {
    arrayToVectorUdf(v)
  }
}
