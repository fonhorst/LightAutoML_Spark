package org.apache.spark.sql.lightautoml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Column
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateFunction, ApproximatePercentile}
import org.apache.spark.sql.functions.udf

object functions {
  private val arrayToVectorUdf = udf { array: Seq[Double] =>
    Vectors.dense(array.toArray)
  }

  private def withAggregateFunction(
                                           func: AggregateFunction,
                                           isDistinct: Boolean = false): Column = {
    Column(func.toAggregateExpression(isDistinct))
  }

  /**
   * Converts a column of array of numeric type into a column of dense vectors in MLlib.
   *
   * @param v : the column of array&lt;NumericType&gt type
   * @return a column of type `org.apache.spark.ml.linalg.Vector`
   * @since 3.1.0
   */
  def array_to_vector(v: Column): Column = {
    arrayToVectorUdf(v)
  }

  def percentile_approx(e: Column, percentage: Column, accuracy: Column): Column = {
    withAggregateFunction {
      new ApproximatePercentile(
        e.expr, percentage.expr, accuracy.expr
      )
    }
  }
}
