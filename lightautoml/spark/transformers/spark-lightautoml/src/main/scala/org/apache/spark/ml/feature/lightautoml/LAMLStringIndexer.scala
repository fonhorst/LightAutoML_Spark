package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.SparkException
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerAggregator, StringIndexerModel}
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.expressions.{If, Literal}
import org.apache.spark.sql.functions.{collect_set, udf}
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Column, DataFrame, Dataset, Encoder, Encoders}
import org.apache.spark.util.ThreadUtils
import org.apache.spark.util.collection.OpenHashMap

@Since("1.4.0")
class LAMLStringIndexer @Since("1.4.0")(
                                               @Since("1.4.0") override val uid: String
                                       ) extends StringIndexer {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("strIdx"))

  @Since("3.2.0")
  val minFreq: Param[Int] = new Param[Int](this, "minFreq", doc = "minFreq")

  /** @group setParam */
  @Since("3.2.0")
  def setMinFreq(value: Int): this.type = set(minFreq, value)

  @Since("3.2.0")
  val defaultValue: Param[Double] = new Param[Double](this, "defaultValue", doc = "defaultValue")

  /** @group setParam */
  @Since("3.2.0")
  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  setDefault(minFreq -> 5, defaultValue -> 0.0F)

  private def getSelectedCols(dataset: Dataset[_], inputCols: Seq[String]): Seq[Column] = {
    inputCols.map { colName =>
      val col = dataset.col(colName)
      if (col.expr.dataType == StringType) {
        col
      } else {
        // We don't count for NaN values. Because `StringIndexerAggregator` only processes strings,
        // we replace NaNs with null in advance.
        new Column(If(col.isNaN.expr, Literal(null), col.expr)).cast(StringType)
      }
    }
  }

  private def countByValue(dataset: Dataset[_],
                           inputCols: Array[String]): Array[OpenHashMap[String, Long]] = {

    val aggregator = new StringIndexerAggregator(inputCols.length)
    implicit val encoder: Encoder[Array[OpenHashMap[String, Long]]] = Encoders.kryo[Array[OpenHashMap[String, Long]]]

    val selectedCols = getSelectedCols(dataset, inputCols)
    dataset.select(selectedCols: _*)
            .toDF
            .groupBy().agg(aggregator.toColumn)
            .as[Array[OpenHashMap[String, Long]]]
            .collect()(0)
  }

  private def sortByFreq(dataset: Dataset[_], ascending: Boolean): Array[Array[String]] = {
    val (inputCols, _) = getInOutCols()

    val sortFunc = StringIndexer.getSortFunc(ascending = ascending)
    val orgStrings = countByValue(dataset, inputCols).toSeq
    ThreadUtils.parmap(orgStrings, "sortingStringLabels", 8) { counts =>
      counts.toSeq.filter(_._2 > $(minFreq)).sortWith(sortFunc).map(_._1).toArray
    }.toArray
  }

  private def sortByAlphabet(dataset: Dataset[_], ascending: Boolean): Array[Array[String]] = {
    val (inputCols, _) = getInOutCols()

    val selectedCols = getSelectedCols(dataset, inputCols).map(collect_set)
    val allLabels = dataset.select(selectedCols: _*)
            .collect().toSeq.flatMap(_.toSeq)
            .asInstanceOf[scala.collection.Seq[scala.collection.Seq[String]]].toSeq
    ThreadUtils.parmap(allLabels, "sortingStringLabels", 8) { labels =>
      val sorted = labels.filter(_ != null).sorted
      if (ascending) {
        sorted.toArray
      } else {
        sorted.reverse.toArray
      }
    }.toArray
  }

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): StringIndexerModel = {
    transformSchema(dataset.schema, logging = true)

    // In case of equal frequency when frequencyDesc/Asc, the strings are further sorted
    // alphabetically.
    val labelsArray = $(stringOrderType) match {
      case StringIndexer.frequencyDesc => sortByFreq(dataset, ascending = false)
      case StringIndexer.frequencyAsc => sortByFreq(dataset, ascending = true)
      case StringIndexer.alphabetDesc => sortByAlphabet(dataset, ascending = false)
      case StringIndexer.alphabetAsc => sortByAlphabet(dataset, ascending = true)
    }
    copyValues(
      new LAMLStringIndexerModel(
        uid = uid,
        labelsArray = labelsArray
      ).setDefaultValue($(defaultValue)).setParent(this)
    )
  }

}


@Since("1.6.0")
object LAMLStringIndexer extends DefaultParamsReadable[LAMLStringIndexer] {
  private[feature] val SKIP_INVALID: String = "skip"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)
  private[feature] val frequencyDesc: String = "frequencyDesc"
  private[feature] val frequencyAsc: String = "frequencyAsc"
  private[feature] val alphabetDesc: String = "alphabetDesc"
  private[feature] val alphabetAsc: String = "alphabetAsc"
  private[feature] val supportedStringOrderType: Array[String] =
    Array(frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc)

  @Since("1.6.0")
  override def load(path: String): LAMLStringIndexer = super.load(path)

  // Returns a function used to sort strings by frequency (ascending or descending).
  // In case of equal frequency, it sorts strings by alphabet (ascending).
  private[feature] def getSortFunc(ascending: Boolean): ((String, Long), (String, Long)) => Boolean = {
    if (ascending) {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (freqA == freqB) {
          strA < strB
        } else {
          freqA < freqB
        }
    } else {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (freqA == freqB) {
          strA < strB
        } else {
          freqA > freqB
        }
    }
  }
}


@Since("1.4.0")
class LAMLStringIndexerModel(override val uid: String,
                             override val labelsArray: Array[Array[String]])
        extends StringIndexerModel(labelsArray) {


  @Since("1.5.0")
  def this(uid: String, labels: Array[String]) = this(uid, Array(labels))

  @Since("1.5.0")
  def this(labels: Array[String]) = this(Identifiable.randomUID("strIdx"), Array(labels))

  @Since("3.0.0")
  def this(labelsArray: Array[Array[String]]) = this(Identifiable.randomUID("strIdx"), labelsArray)
  
  @Since("3.2.0")
  val minFreq: Param[Int] = new Param[Int](this, "minFreq", doc = "minFreq")

  /** @group setParam */
  @Since("3.2.0")
  def setMinFreq(value: Int): this.type = set(minFreq, value)

  @Since("3.2.0")
  val defaultValue: Param[Double] = new Param[Double](this, "defaultValue", doc = "defaultValue")

  /** @group setParam */
  @Since("3.2.0")
  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  setDefault(minFreq -> 5, defaultValue -> 0.0D)


  // Prepares the maps for string values to corresponding index values.
  private val labelsToIndexArray: Array[OpenHashMap[String, Double]] = {
    for (labels <- labelsArray) yield {
      val n = labels.length
      val map = new OpenHashMap[String, Double](n)
      labels.zipWithIndex.foreach { case (label, idx) =>
        map.update(label, idx + 1)
      }
      map
    }
  }

  // This filters out any null values and also the input labels which are not in
  // the dataset used for fitting.
  private def filterInvalidData(dataset: Dataset[_], inputColNames: Seq[String]): Dataset[_] = {
    val conditions: Seq[Column] = (0 until inputColNames.length).map { i =>
      val inputColName = inputColNames(i)
      val labelToIndex = labelsToIndexArray(i)
      // We have this additional lookup at `labelToIndex` when `handleInvalid` is set to
      // `StringIndexer.SKIP_INVALID`. Another idea is to do this lookup natively by SQL
      // expression, however, lookup for a key in a map is not efficient in SparkSQL now.
      // See `ElementAt` and `GetMapValue` expressions. If SQL's map lookup is improved,
      // we can consider to change this.
      val filter = udf { label: String =>
        labelToIndex.contains(label)
      }
      filter(dataset(inputColName))
    }

    dataset.na.drop(inputColNames.filter(dataset.schema.fieldNames.contains(_)))
            .where(conditions.reduce(_ and _))
  }

  private def getIndexer(labelToIndex: OpenHashMap[String, Double]) = {
    val keepInvalid = (getHandleInvalid == StringIndexer.KEEP_INVALID)

    udf { label: String =>
      if (label == null) {
        if (keepInvalid) {
          $(defaultValue)
        } else {
          throw new SparkException("StringIndexer encountered NULL value. To handle or skip " +
                  "NULLS, try setting StringIndexer.handleInvalid.")
        }
      } else {
        if (labelToIndex.contains(label)) {
          labelToIndex(label)
        } else if (keepInvalid) {
          $(defaultValue)
        } else {
          throw new SparkException(s"Unseen label: $label. To handle unseen labels, " +
                  s"set Param handleInvalid to ${StringIndexer.KEEP_INVALID}.")
        }
      }
    }.asNondeterministic()
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val (inputColNames, outputColNames) = getInOutCols()
    val outputColumns = new Array[Column](outputColNames.length)

    // Skips invalid rows if `handleInvalid` is set to `StringIndexer.SKIP_INVALID`.
    val filteredDataset = if (getHandleInvalid == StringIndexer.SKIP_INVALID) {
      filterInvalidData(dataset, inputColNames)
    } else {
      dataset
    }

    for (i <- 0 until outputColNames.length) {
      val inputColName = inputColNames(i)
      val outputColName = outputColNames(i)
      val labelToIndex = labelsToIndexArray(i)
      val labels = labelsArray(i)

      if (!dataset.schema.fieldNames.contains(inputColName)) {
        logWarning(s"Input column ${inputColName} does not exist during transformation. " +
                "Skip StringIndexerModel for this column.")
        outputColNames(i) = null
      } else {
        val filteredLabels = getHandleInvalid match {
          case StringIndexer.KEEP_INVALID => labels :+ "__unknown"
          case _ => labels
        }
        val metadata = NominalAttribute.defaultAttr
                .withName(outputColName)
                .withValues(filteredLabels)
                .toMetadata()

        val indexer = getIndexer(labelToIndex)

        outputColumns(i) = indexer(dataset(inputColName).cast(StringType))
                .as(outputColName, metadata)
      }
    }

    val filteredOutputColNames = outputColNames.filter(_ != null)
    val filteredOutputColumns = outputColumns.filter(_ != null)

    require(filteredOutputColNames.length == filteredOutputColumns.length)
    if (filteredOutputColNames.length > 0) {
      filteredDataset.withColumns(filteredOutputColNames, filteredOutputColumns)
    } else {
      filteredDataset.toDF()
    }
  }
}
