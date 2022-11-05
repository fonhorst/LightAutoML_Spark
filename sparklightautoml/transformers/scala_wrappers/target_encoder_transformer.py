from typing import Dict, List

from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer

from sparklightautoml.mlwriters import CommonJavaToPythonMLReadable


@inherit_doc
class TargetEncoderTransformer(JavaTransformer, HasInputCols, HasOutputCols,
                               CommonJavaToPythonMLReadable, JavaMLWritable):
    """
    Scala-based implementation of Target Encoder transformer
    """

    def __init__(self, *,
                 enc: Dict[str, List[float]],
                 oof_enc: Dict[str, List[List[float]]],
                 fold_column: str,
                 apply_oof: bool,
                 input_cols: List[str],
                 output_cols: List[str]
                 ):
        super(TargetEncoderTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.TargetEncoderTransformer",
            self.uid, enc, oof_enc, fold_column, apply_oof
        )

        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, output_cols)

    def setInputCols(self, value) -> 'TargetEncoderTransformer':
        self.set(self.inputCols, value)
        return self

    def setOutputCols(self, value) -> 'TargetEncoderTransformer':
        self.set(self.outputCols, value)
        return self
