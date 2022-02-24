package org.apache.spark.ml.feature.lightautoml

import org.apache.hadoop.fs.Path

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.annotation.Since
import org.apache.spark.ml.util._
import org.apache.spark.util.VersionUtils.majorMinorVersion

@Since("1.6.0")
object LAMLStringIndexer extends DefaultParamsReadable[StringIndexer] {
    private[feature] val SKIP_INVALID: String = "skip"
    private[feature] val ERROR_INVALID: String = "error"
    private[feature] val KEEP_INVALID: String = "keep"
    private[feature] val supportedHandleInvalids: Array[String] =
        Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)
    private[feature] val frequencyDesc: String = "frequencyDesc"
    private[feature] val frequencyAsc: String =  "frequencyAsc"
    private[feature] val alphabetDesc: String = "alphabetDesc"
    private[feature] val alphabetAsc: String = "alphabetAsc"
    private[feature] val supportedStringOrderType: Array[String] =
        Array(frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc)

    @Since("1.6.0")
    override def load(path: String): StringIndexer = super.load(path)

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


@Since("1.6.0")
object LAMLStringIndexerModel extends MLReadable[StringIndexerModel] {

    private[LAMLStringIndexerModel]
    class StringIndexModelWriter(instance: StringIndexerModel) extends MLWriter {

        private case class Data(labelsArray: Array[Array[String]])

        override protected def saveImpl(path: String): Unit = {
            DefaultParamsWriter.saveMetadata(instance, path, sc)
            val data = Data(instance.labelsArray)
            val dataPath = new Path(path, "data").toString
            sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
        }
    }

    private class StringIndexerModelReader extends MLReader[StringIndexerModel] {

        private val className = classOf[StringIndexerModel].getName

        override def load(path: String): StringIndexerModel = {
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
                val labels = data.getAs[Seq[String]](0).toArray
                Array(labels)
            } else {
                // After Spark 3.0.
                val data = sparkSession.read.parquet(dataPath)
                        .select("labelsArray")
                        .head()
                data.getSeq[scala.collection.Seq[String]](0).map(_.toArray).toArray
            }
            val model = new StringIndexerModel(metadata.uid, labelsArray)
            metadata.getAndSetParams(model)
            model
        }
    }

    @Since("1.6.0")
    override def read: MLReader[StringIndexerModel] = new StringIndexerModelReader

    @Since("1.6.0")
    override def load(path: String): StringIndexerModel = super.load(path)
}