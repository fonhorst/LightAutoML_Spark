from typing import Optional

import numpy as np
from pandas import Series
from pyspark.sql import functions as F

from lightautoml.dataset.roles import CategoryRole
from lightautoml.transformers.categorical import categorical_check

from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer


class LabelEncoder(SparkTransformer):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    def __init__(self, *args, **kwargs):
        self._output_role = CategoryRole(np.int32, label_encoded=True)

    def fit(self, dataset: SparkDataset) -> "LabelEncoder":
        super().fit(dataset)
        roles = dataset.roles

        cached_dataset = dataset.data.cache()

        self.dicts = {}
        for i in cached_dataset.columns:
            role = roles[i]

            # TODO: think what to do with this warning
            co = role.unknown

            # FIXME: Possible OOM point
            vals = cached_dataset\
                .groupBy(i).count() \
                .filter(F.col("count") > co)\
                .orderBy("count", ascending=False)\
                .select(i)\
                .toPandas()

            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])

        cached_dataset.unpersist()

        return self

