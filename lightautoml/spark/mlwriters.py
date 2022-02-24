import json
import logging
import os
import pickle
import time

from pathlib import Path

from pyspark.ml.util import MLReadable
from pyspark.ml.util import MLReader
from pyspark.ml.util import MLWritable
from pyspark.ml.util import MLWriter


logger = logging.getLogger(__name__)


class CommonPickleMLWritable(MLWritable):
    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return СommonPickleMLWriter(self)


class CommonPickleMLReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return СommonPickleMLReader()


class СommonPickleMLWriter(MLWriter):
    """Implements saving an Estimator/Transformer instance to disk.
    Used when saving a trained pipeline.
    Implements MLWriter.saveImpl(path) method.
    """

    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        logger.info(f"Save {self.instance.__class__.__name__} to {path}")

        СommonPickleMLWriter.saveMetadata(self.instance, path, self.sc)

        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "transformer_class_instance.pickle"), 'wb') as handle:
            pickle.dump(self.instance, handle)

    @staticmethod
    def saveMetadata(instance, path, sc):
        """
        Saves metadata + Params to: path + "/metadata"

        - class
        - timestamp
        - sparkVersion
        - uid
        - paramMap
        - defaultParamMap (since 2.4.0)
        - (optionally, extra metadata)

        Parameters
        ----------
        extraMetadata : dict, optional
            Extra metadata to be saved at same level as uid, paramMap, etc.
        paramMap : dict, optional
            If given, this is saved in the "paramMap" field.
        """
        metadataPath = os.path.join(path, "metadata")
        metadataJson = СommonPickleMLWriter._get_metadata_to_save(instance,
                                                                  sc)
        sc.parallelize([metadataJson], 1).saveAsTextFile(metadataPath)

    @staticmethod
    def _get_metadata_to_save(instance, sc):
        """
        Helper for :py:meth:`СommonPickleMLWriter.saveMetadata` which extracts the JSON to save.
        This is useful for ensemble models which need to save metadata for many sub-models.

        Notes
        -----
        See :py:meth:`DefaultParamsWriter.saveMetadata` for details on what this includes.
        """
        uid = instance.uid
        cls = instance.__module__ + '.' + instance.__class__.__name__

        basicMetadata = {"class": cls, "timestamp": int(round(time.time() * 1000)),
                         "sparkVersion": sc.version, "uid": uid, "paramMap": None,
                         "defaultParamMap": None}

        return json.dumps(basicMetadata, separators=[',', ':'])


class СommonPickleMLReader(MLReader):

    def load(self, path):
        """Load the ML instance from the input path."""

        with open(os.path.join(path, "transformer_class_instance.pickle"), 'rb') as handle:
            instance = pickle.load(handle)

        return instance
