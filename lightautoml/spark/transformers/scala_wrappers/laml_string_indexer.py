# from pyspark.ml.feature import StringIndexer

from pyspark import since, keyword_only, SparkContext
from pyspark.ml.param.shared import HasThreshold, HasThresholds, HasInputCol, HasOutputCol, \
    HasInputCols, HasOutputCols, HasHandleInvalid, HasRelativeError, HasFeaturesCol, HasLabelCol, \
    HasSeed, HasNumFeatures, HasStepSize, HasMaxIter, TypeConverters, Param, Params
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaTransformer


class _StringIndexerParams(JavaParams, HasHandleInvalid, HasInputCol, HasOutputCol,
                           HasInputCols, HasOutputCols):
    """
    Params for :py:class:`StringIndexer` and :py:class:`StringIndexerModel`.
    """

    stringOrderType = Param(Params._dummy(), "stringOrderType",
                            "How to order labels of string column. The first label after " +
                            "ordering is assigned an index of 0. Supported options: " +
                            "frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc. " +
                            "Default is frequencyDesc. In case of equal frequency when " +
                            "under frequencyDesc/Asc, the strings are further sorted " +
                            "alphabetically",
                            typeConverter=TypeConverters.toString)

    handleInvalid = Param(Params._dummy(), "handleInvalid", "how to handle invalid data (unseen " +
                          "or NULL values) in features and label column of string type. " +
                          "Options are 'skip' (filter out rows with invalid data), " +
                          "error (throw an error), or 'keep' (put invalid data " +
                          "in a special additional bucket, at index numLabels).",
                          typeConverter=TypeConverters.toString)

    minFreq = Param(Params._dummy(),
                    "minFreq",
                    "The minimum number of the element occurrences not to skip it.",
                    typeConverter=TypeConverters.toInt)

    defaultValue = Param(Params._dummy(),
                         "defaultValue",
                         "The index for unknown labels and labels that are too rare.",
                         typeConverter=TypeConverters.toFloat)

    def __init__(self, *args):
        super(_StringIndexerParams, self).__init__(*args)
        self._setDefault(
            handleInvalid="error",
            stringOrderType="frequencyDesc",
            minFreq=5,
            defaultValue=0.
        )

    @since("2.3.0")
    def getStringOrderType(self):
        """
        Gets the value of :py:attr:`stringOrderType` or its default value 'frequencyDesc'.
        """
        return self.getOrDefault(self.stringOrderType)


class LAMLStringIndexer(JavaEstimator, _StringIndexerParams, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, *, inputCol=None, outputCol=None, inputCols=None, outputCols=None,
                 handleInvalid="error", stringOrderType="frequencyDesc", minFreq=5, defaultValue=0.):
        """
        __init__(self, \\*, inputCol=None, outputCol=None, inputCols=None, outputCols=None, \
                 handleInvalid="error", stringOrderType="frequencyDesc")
        """
        super(LAMLStringIndexer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexer",
            self.uid,
            self.minFreq,
            self.defaultValue
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, *, inputCol=None, outputCol=None, inputCols=None, outputCols=None,
                  handleInvalid="error", stringOrderType="frequencyDesc", minFreq=5, defaultValue=0.):

        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return LAMLStringIndexerModel(java_model)

    @since("2.3.0")
    def setStringOrderType(self, value):
        """
        Sets the value of :py:attr:`stringOrderType`.
        """
        return self._set(stringOrderType=value)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    @since("3.0.0")
    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    @since("3.0.0")
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)

    def setHandleInvalid(self, value):
        """
        Sets the value of :py:attr:`handleInvalid`.
        """
        return self._set(handleInvalid=value)

    @since("3.2.0")
    def setMinFreq(self, value):
        """
        Sets the value of :py:attr:`minFreq`.
        """
        return self._set(minFreq=value)

    @since("3.2.0")
    def setDefaultValue(self, value):
        """
        Sets the value of :py:attr:`defaultValue`.
        """
        return self._set(defaultValue=value)


class LAMLStringIndexerModel(JavaModel, _StringIndexerParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`StringIndexer`.

    .. versionadded:: 1.4.0
    """

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    @since("3.0.0")
    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    @since("3.0.0")
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)

    @since("2.4.0")
    def setHandleInvalid(self, value):
        """
        Sets the value of :py:attr:`handleInvalid`.
        """
        return self._set(handleInvalid=value)

    @since("3.2.0")
    def setDefaultValue(self, value):
        """
        Sets the value of :py:attr:`defaultValue`.
        """
        return self._set(defaultValue=value)

    @classmethod
    @since("2.4.0")
    def from_labels(cls, labels, inputCol, outputCol=None, handleInvalid=None, defaultValue=0.):
        """
        Construct the model directly from an array of label strings,
        requires an active SparkContext.
        """
        sc = SparkContext._active_spark_context
        java_class = sc._gateway.jvm.java.lang.String
        jlabels = LAMLStringIndexerModel._new_java_array(labels, java_class)
        model = LAMLStringIndexerModel._create_from_java_class(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel",
            jlabels,
            defaultValue
        )
        model.setInputCol(inputCol)
        if outputCol is not None:
            model.setOutputCol(outputCol)
        if handleInvalid is not None:
            model.setHandleInvalid(handleInvalid)
        return model

    @classmethod
    @since("3.0.0")
    def from_arrays_of_labels(cls, arrayOfLabels, inputCols, outputCols=None,
                              handleInvalid=None, defaultValue=0.):
        """
        Construct the model directly from an array of array of label strings,
        requires an active SparkContext.
        """
        sc = SparkContext._active_spark_context
        java_class = sc._gateway.jvm.java.lang.String
        jlabels = LAMLStringIndexerModel._new_java_array(arrayOfLabels, java_class)
        model = LAMLStringIndexerModel._create_from_java_class(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel",
            jlabels,
            defaultValue
        )
        model.setInputCols(inputCols)
        if outputCols is not None:
            model.setOutputCols(outputCols)
        if handleInvalid is not None:
            model.setHandleInvalid(handleInvalid)
        return model

    @property
    @since("1.5.0")
    def labels(self):
        """
        Ordered list of labels, corresponding to indices to be assigned.

        .. deprecated:: 3.1.0
            It will be removed in future versions. Use `labelsArray` method instead.
        """
        return self._call_java("labels")

    @property
    @since("3.0.2")
    def labelsArray(self):
        """
        Array of ordered list of labels, corresponding to indices to be assigned
        for each input column.
        """
        return self._call_java("labelsArray")
