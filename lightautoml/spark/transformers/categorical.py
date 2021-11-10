from typing import Optional
from itertools import chain
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

            # FIXME (Distr-LAMA): Possible OOM point
            vals = cached_dataset \
                .groupBy(i).count() \
                .filter(F.col("count") > co) \
                .orderBy(["count", i], ascending=[False, True]) \
                .select(i) \
                .toPandas()

            # FIXME (Distr-LAMA): Do we really need collecting this data? It is used in transform method and
            # it may be joined. I propose to keep this variable as a spark dataframe. Discuss?
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])

        cached_dataset.unpersist()

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        df = dataset.data

        for i in df.columns:

            # Since defaultdict is not working with the na.replace (its behaviour is the same as simple dict
            # i.e. if there is no key - it just skips and leave the value, I had to introduce this crutch
            # of the map column.
            # The most possible elegant way (but it is not working):
            # df.na.replace(defaultdict(lambda: self._fillna_val, self.dicts[i].to_dict()), subset=[i])....
            labels = F.create_map([F.lit(x) for x in chain(*self.dicts[i].to_dict().items())])
            df = df \
                .withColumn(i, labels[df[i]]) \
                .fillna(self._fillna_val) \
                .withColumn(i, F.col(i).cast("int")) \
                .withColumnRenamed(i, f"{self._fname_prefix}__{i}")
                # FIXME (Distr-LAMA): Probably we have to write a converter numpy/python/pandas types => spark types?

        output: SparkDataset = dataset.empty()
        output.set_data(df, self.features, self._output_role)

        return output
