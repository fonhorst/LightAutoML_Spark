from typing import Dict, List
from uuid import uuid4

from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaParams

from sparklightautoml.mlwriters import CommonJavaToPythonMLReadable


@inherit_doc
class TargetEncoderTransformer(JavaTransformer, HasInputCols, HasOutputCols,
                               CommonJavaToPythonMLReadable, JavaMLWritable):
    """
    Scala-based implementation of Target Encoder transformer
    """

    @classmethod
    def create(cls, *,
               enc: Dict[str, List[float]],
               oof_enc: Dict[str, List[List[float]]],
               fold_column: str,
               apply_oof: bool,
               input_cols: List[str],
               output_cols: List[str]):
        uid = f"TargetEncoderTransformer_{str(uuid4()).replace('-', '_')}"
        _java_obj = cls._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.TargetEncoderTransformer",
            uid, enc, oof_enc, fold_column, apply_oof
        )

        tet = TargetEncoderTransformer(_java_obj).setInputCols(input_cols).setOutputCols(output_cols)
        return tet

    def __init__(self,java_obj):
        super(TargetEncoderTransformer, self).__init__()
        self._java_obj = java_obj

    def setInputCols(self, value) -> 'TargetEncoderTransformer':
        self.set(self.inputCols, value)
        return self

    def setOutputCols(self, value) -> 'TargetEncoderTransformer':
        self.set(self.outputCols, value)
        return self

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.

        Meta-algorithms such as Pipeline should override this method as a classmethod.
        """

        def __get_class(clazz):
            """
            Loads Python class from its name.
            """
            parts = clazz.split(".")
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m

        stage_name = "sparklightautoml.transformers.scala_wrappers.target_encoder_transformer.TargetEncoderTransformer"
        # Generate a default new instance from the stage_name class.
        py_type = __get_class(stage_name)
        if issubclass(py_type, JavaParams):
            # Load information from java_stage to the instance.
            py_stage = py_type(java_stage)
            # py_stage._java_obj = java_stage
            py_stage._resetUid(java_stage.uid())
            py_stage._transfer_params_from_java()
        elif hasattr(py_type, "_from_java"):
            py_stage = py_type._from_java(java_stage)
        else:
            raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r" % stage_name)
        return py_stage
