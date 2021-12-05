from abc import ABC
from typing import Optional, cast, Tuple, List, Sequence

from pyspark.sql.types import FloatType

from lightautoml.automl.blend import Blender, \
    BestModelSelector as LAMABestModelSelector, \
    WeightedBlender as LAMAWeightedBlender
from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.dataset.base import SparkDataset

from pyspark.sql import functions as F

import numpy as np

from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole


class BlenderMixin(Blender, ABC):
    pass
    # def score(self, dataset: LAMLDataset) -> float:
    #     # TODO: SPARK-LAMA convert self._score to a required metric
    #
    #     raise NotImplementedError()


class BestModelSelector(BlenderMixin, LAMABestModelSelector):
    pass


class WeightedBlender(BlenderMixin, LAMAWeightedBlender):
    def _get_weighted_pred(self, splitted_preds: Sequence[SparkDataset], wts: Optional[np.ndarray]) -> SparkDataset:
        length = len(splitted_preds)
        if wts is None:
            # wts = np.ones(length, dtype=np.float32) / length
            wts = [1.0 / length for _ in range(length)]
        else:
            wts = [float(el) for el in wts]

        assert len(splitted_preds[0].features) == 1, \
            f"There should be only one feature containing predictions in the form of array, " \
            f"but: {splitted_preds[0].features}"

        feat = splitted_preds[0].features[0]

        assert isinstance(splitted_preds[0].roles[feat], NumericVectorOrArrayRole), \
            f"The prediction should be an array or vector, but {type(splitted_preds[0].roles[feat])}"

        role = splitted_preds[0].roles[feat]

        nan_feat = f"{feat}_nan_conf"

        sdfs = [
            x.data.select(
                SparkDataset.ID_COLUMN,
                F.transform(feat, lambda x: F.when(F.isnan(x), 0.0).otherwise(x * w)).alias(feat),
                F.when(F.array_contains(feat, float('nan')), 0.0).otherwise(w).alias(nan_feat)
            )
            for (x, w) in zip(splitted_preds, wts)
        ]

        sum_sdf = sdfs[0]
        for sdf in sdfs[1:]:
            summing_pred_arrays = \
                F.transform(F.arrays_zip(sum_sdf[feat], sdf[feat]), lambda x, y: x + y)
            sum_sdf = (
                sum_sdf
                .join(sdf, on=SparkDataset.ID_COLUMN)
                .select(
                    sum_sdf[SparkDataset.ID_COLUMN],
                    summing_pred_arrays,
                    (sum_sdf[nan_feat] + sdf[nan_feat]).alias(nan_feat)
                )
            )

        wfeat_name = f"WeightedBlend_{feat}"

        # TODO: SPARK-LAMA potentially this is a bad place check it later:
        #  1. equality condition double types
        #  2. None instead of nan (in the origin)
        #  due to Spark doesn't allow to mix types in the same column
        weighted_sdf = sum_sdf.select(
            SparkDataset.ID_COLUMN,
            F.when(F.col(nan_feat) == 0.0, None)
                .otherwise(F.transform(feat, lambda x: x / F.col(nan_feat)))
                .alias(wfeat_name)
        )

        output = splitted_preds[0].empty()
        output.set_data(weighted_sdf, [wfeat_name], role)

        return output
