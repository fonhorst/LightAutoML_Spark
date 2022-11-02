package org.apache.spark.ml.feature.lightautoml

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel.LAMLStringIndexModelWriter
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, DefaultParamsWriter, MLReadable, MLReader, MLWriter}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{IntegerType, ShortType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object TargetEncoderTransformer extends MLReadable[TargetEncoderTransformer] {
  type Encodings = Map[String, Array[Double]]
  type OofEncodings = Map[String, Array[Array[Double]]]

  private[TargetEncoderTransformer] class TargetEncoderTransformerWriter(instance: TargetEncoderTransformer)
          extends MLWriter {

    private case class Data(labelsArray: Array[Array[(String, Long)]])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.labelsArray)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class TargetEncoderTransformerReader extends MLReader[TargetEncoderTransformer] {

    private val className = classOf[TargetEncoderTransformer].getName

    override def load(path: String): TargetEncoderTransformer = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString

      // We support loading old `StringIndexerModel` saved by previous Spark versions.
      // Previous model has `labels`, but new model has `labelsArray`.
      val (majorVersion, minorVersion) = majorMinorVersion(metadata.sparkVersion)
      val labelsArray = if (majorVersion < 3) {
        // Spark 2.4 and before.
        val data = sparkSession.read.parquet(dataPath)
                .select("labels")
                .head()
        val labels = data.getAs[Seq[(String, Long)]](0).toArray
        Array(labels)
      } else {
        // After Spark 3.0.
        val data = sparkSession.read.parquet(dataPath)
                .select("labelsArray")
                .head()

        val res = data.getSeq[scala.collection.Seq[GenericRowWithSchema]](0)
        res.map(_.map(x => (x.getAs[String](0), x.getAs[Long](1))).toArray).toArray
      }
      val model = new LAMLStringIndexerModel(metadata.uid, labelsArray)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[TargetEncoderTransformer] = new TargetEncoderTransformerReader

  override def load(path: String): TargetEncoderTransformer = super.load(path)
}

// encodings - (column, cat_seq_id -> value)
// oof_encodings - (columns, fold_id -> cat_seq_id -> value)
class TargetEncoderTransformer(override val uid: String,
                               enc: TargetEncoderTransformer.Encodings,
                               oof_enc: TargetEncoderTransformer.OofEncodings,
                               fold_column: String)
        extends Transformer
                with HasInputCols
                with HasOutputCols
                with DefaultParamsWritable
                with DefaultParamsReadable[TargetEncoderTransformer] {

  import TargetEncoderTransformer._

  val encodings: Param[Encodings] = new Param[Encodings](
    this, "encodings",
    "Encodings to be applied during normal transform",
    (_:Encodings) => true
  )

  val oofEncodings: Param[OofEncodings] = new Param[OofEncodings](
    this, "oofEncodings",
    "Encodings taking care of folds to be applied during fit_transform only",
    (_:OofEncodings) => true
  )

  val applyOof: Param[Boolean] = new Param[Boolean](
    this, "applyOof",
    "Apply oof encodings instead of just encodings",
    (_: Boolean) => true
  )

  val foldColumn: Param[String] = new Param[String](
    this, "foldColumn",
    "Fold column name to be used when applying oof encodings",
    (_: String) => true
  )

  this.set(encodings, enc)
  this.set(oofEncodings, oof_enc)
  this.set(applyOof, true)
  this.set(foldColumn, fold_column)

  def setEncodings(enc: Encodings): this.type = set(encodings, enc)

  def getEncodings: Option[Encodings] = get(encodings)

  def setOofEncodings(oof_enc: OofEncodings): this.type = set(oofEncodings, oof_enc)

  def getOofEncodings: Option[OofEncodings] = get(oofEncodings)

  def setApplyOof(oof: Boolean): this.type = set(applyOof, oof)

  def getApplyOof: Option[Boolean] = get(applyOof)

  def setFoldColumn(col: String): this.type = set(foldColumn, col)

  def getFoldColumn: Option[String] = get(foldColumn)

  def setInputCols(cols: Array[String]): this.type = set(inputCols, cols)

  def setOutputCols(cols: Array[String]): this.type = set(outputCols, cols)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    transformSchema(dataset.schema)

    val outColumns = getApplyOof match {
      case Some(true) if getOofEncodings.isEmpty =>
        throw new IllegalArgumentException("OofEncodings cannot be unset if applyOof is true")
      case Some(true) if getFoldColumn.isEmpty =>
        throw new IllegalArgumentException("foldCol cannot be unset if applyOof is true")
      case Some(true) =>
        val oofEncodingsBcst = spark.sparkContext.broadcast(getOofEncodings.get)
        val func = udf((col_name: String, fold: Integer, cat: Integer) => {
          oofEncodingsBcst.value(col_name)(fold)(cat)
        })
        getInputCols.zip(getOutputCols).map{
          case (in_col, out_col) => func(lit(in_col), col(getFoldColumn.get), col(in_col)).alias(out_col)
        }
      case Some(false) if getEncodings.isEmpty =>
        throw new IllegalArgumentException("Encodings cannot be unset if applyOof is false")
      case Some(false) =>
        val encodingsBcst = spark.sparkContext.broadcast(getEncodings.get)
        val func = udf((col_name: String, cat: Integer) => {
          encodingsBcst.value(col_name)(cat)
        })
        getInputCols.zip(getOutputCols).map {
          case (in_col, out_col) => func(lit(in_col), col(in_col)).alias(out_col)
        }
      case None =>
        throw new IllegalArgumentException("applyOof cannot be None")
    }

    dataset.withColumns(getOutputCols, outColumns)
  }

  override def copy(extra: ParamMap): Transformer = {
    val copied = new TargetEncoderTransformer(uid, enc, oof_enc, fold_column)
    copyValues(copied, extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def write: LAMLStringIndexModelWriter = new LAMLStringIndexModelWriter(this)

  override def toString: String = {
    s"StringIndexerModel: uid=$uid, handleInvalid=${$(handleInvalid)}" +
            get(stringOrderType).map(t => s", stringOrderType=$t").getOrElse("") +
            get(inputCols).map(c => s", numInputCols=${c.length}").getOrElse("") +
            get(outputCols).map(c => s", numOutputCols=${c.length}").getOrElse("")
  }

  private def validateAndTransformField(schema: StructType,
                                        inputColName: String,
                                        outputColName: String): StructField = {
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == IntegerType || inputDataType == ShortType,
      s"The input column $inputColName must be integer type or short type, " +
              s"but got $inputDataType.")
    require(schema.fields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    NumericAttribute.defaultAttr.withName(outputColName).toStructField()
  }

  private def validateAndTransformSchema(schema: StructType,
                                         skipNonExistsCol: Boolean = false): StructType = {
    val inputColNames = getInputCols
    val outputColNames = getOutputCols

    require(outputColNames.distinct.length == outputColNames.length,
      s"Output columns should not be duplicate.")

    val outputFields = inputColNames.zip(outputColNames).flatMap {
      case (inputColName, outputColName) =>
        schema.fieldNames.contains(inputColName) match {
          case true => Some(validateAndTransformField(schema, inputColName, outputColName))
          case false if skipNonExistsCol => None
          case _ => throw new SparkException(s"Input column $inputColName does not exist.")
        }
    }
    StructType(schema.fields ++ outputFields)
  }
}


