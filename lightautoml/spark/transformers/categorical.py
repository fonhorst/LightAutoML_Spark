from typing import Optional
from collections import defaultdict
from itertools import chain
import numpy as np
from pandas import Series
from pyspark.sql import functions as F, types as SparkTypes

from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.transformers.categorical import categorical_check

from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer

import copy


class LabelEncoder(SparkTransformer):

    _ad_hoc_types_mapper = defaultdict(
        lambda: "string",
        {
            "bool": "boolean",
            "int": "int",
            "int8": "int",
            "int16": "int",
            "int32": "int",
            "int64": "int",
            "int128": "bigint",
            "int256": "bigint",
            "integer": "int",
            "uint8": "int",
            "uint16": "int",
            "uint32": "int",
            "uint64": "int",
            "uint128": "bigint",
            "uint256": "bigint",
            "longlong": "long",
            "ulonglong": "long",
            "float16": "float",
            "float": "float",
            "float32": "float",
            "float64": "double",
            "float128": "double"
        }
    )

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

            # FIXME SPARK-LAMA: Possible OOM point
            vals = cached_dataset \
                .groupBy(i).count() \
                .filter(F.col("count") > co) \
                .orderBy(["count", i], ascending=[False, True]) \
                .select(i) \
                .toPandas()

            # FIXME SPARK-LAMA: Do we really need collecting this data? It is used in transform method and
            # it may be joined. I propose to keep this variable as a spark dataframe. Discuss?
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])

        cached_dataset.unpersist()

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:

        super().transform(dataset)

        cached_dataset = dataset.data.cache()

        for i in cached_dataset.columns:

            # FIXME SPARK-LAMA: Dirty hot-fix

            role = dataset.roles[i]

            cached_dataset = cached_dataset.withColumn(i, F.col(i).cast(self._ad_hoc_types_mapper[role.dtype.__name__]))

            if i in self.dicts:

                if len(self.dicts[i]) > 0:

                    labels = F.create_map([F.lit(x) for x in chain(*self.dicts[i].to_dict().items())])

                    if np.issubdtype(role.dtype, np.number):
                        cached_dataset = cached_dataset \
                                            .withColumn(i, F.when(F.col(i).isNull(), np.nan)
                                                            .otherwise(F.col(i))
                                                        ) \
                                            .withColumn(i, labels[F.col(i)])
                    else:
                        if None in self.dicts[i].index:
                            cached_dataset = cached_dataset \
                                                .withColumn(i, F.when(F.col(i).isNull(), self.dicts[i][None])
                                                                .otherwise(labels[F.col(i)])
                                                            )
                        else:
                            cached_dataset = cached_dataset \
                                .withColumn(i, labels[F.col(i)])
                else:
                    cached_dataset = cached_dataset \
                        .withColumn(i, F.lit(self._fillna_val))

            cached_dataset = cached_dataset.fillna(self._fillna_val, subset=[i]) \
                .withColumn(i, F.col(i).cast(self._ad_hoc_types_mapper[self._output_role.dtype.__name__])) \
                .withColumnRenamed(i, f"{self._fname_prefix}__{i}")
                # FIXME SPARK-LAMA: Probably we have to write a converter numpy/python/pandas types => spark types?

        output: SparkDataset = dataset.empty()
        output.set_data(cached_dataset, self.features, self._output_role)

        cached_dataset.unpersist()

        return output


class FreqEncoder(LabelEncoder):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    _fillna_val = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def fit(self, dataset: SparkDataset) -> "FreqEncoder":

        super().fit(dataset)

        cached_dataset = dataset.data.cache()

        self.dicts = {}
        for i in cached_dataset.columns:
            vals = cached_dataset \
                .groupBy(i).count() \
                .filter(F.col("count") > 1) \
                .orderBy(["count", i], ascending=[False, True]) \
                .select([i, "count"]) \
                .toPandas()

            self.dicts[i] = vals.set_index(i)["count"]

        cached_dataset.unpersist()

        print(f"FreqFit: {self.dicts}")

        return self


class OrdinalEncoder(LabelEncoder):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "ord"

    _spark_numeric_types = [
        SparkTypes.ByteType,
        SparkTypes.ShortType,
        SparkTypes.IntegerType,
        SparkTypes.LongType,
        SparkTypes.FloatType,
        SparkTypes.DoubleType,
        SparkTypes.DecimalType
    ]

    _fillna_val = np.nan

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def fit(self, dataset: SparkDataset):

        super().fit(dataset)

        # convert to accepted dtype and get attributes
        roles = dataset.roles

        cached_dataset = dataset.data.cache()

        self.dicts = {}
        for i in cached_dataset.columns:
            role = roles[i]

            if not type(cached_dataset.schema[i].dataType) in self._spark_numeric_types:

                co = role.unknown

                cnts = cached_dataset \
                    .groupBy(i).count() \
                    .filter(F.col("count") > co) \
                    .na.replace(np.nan, None) \
                    .filter(F.col(i).isNotNull()) \
                    .select(i) \
                    .toPandas()

                cnts = Series(cnts[i].astype(str).rank().values, index=cnts[i])
                self.dicts[i] = cnts.append(Series([cnts.shape[0] + 1], index=[np.nan])).drop_duplicates()

        cached_dataset.unpersist()
        print(f"OrdinalFit SPRK: {self.dicts}")

        return self
