from abc import ABC
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
from pyspark.sql import functions as F

from lightautoml.automl.blend import Blender, \
    BestModelSelector as LAMABestModelSelector, \
    WeightedBlender as LAMAWeightedBlender
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline

from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasOutputCols


class SparkBlender:
    """Basic class for blending.

    Blender learns how to make blend
    on sequence of prediction datasets and prune pipes,
    that are not used in final blend.

    """

    @property
    def transformer(self) -> Transformer:
        """Returns Spark MLlib Transformer.
        Represents a Transformer with fitted models."""

        assert self._transformer is not None, "Pipeline is not fitted!"

        return self._transformer

    def fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        """Wraps custom ``._fit_predict`` methods of blenders.

        Method wraps individual ``._fit_predict`` method of blenders.
        If input is single model - take it, else ``._fit_predict``
        Note - some pipelines may have more than 1 model.
        So corresponding prediction dataset have multiple prediction cols.

        Args:
            predictions: Sequence of datasets with predictions.
            pipes: Sequence of pipelines.

        Returns:
            Single prediction dataset and sequence of pruned pipelines.

        """
        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            self._bypass = True
            return predictions, pipes

        return self._fit_predict(predictions, pipes)


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

        assert len(splitted_preds[0].features) == 1, \
            f"There should be only one feature containing predictions in the form of array, " \
            f"but: {splitted_preds[0].features}"

        feat = splitted_preds[0].features[0]
        role = splitted_preds[0].roles[feat]
        task = splitted_preds[0].task
        nan_feat = f"{feat}_nan_conf"
        # we put 0 here, because there cannot be more than output
        # even if works with vectors
        wfeat_name = "WeightedBlend_0"
        length = len(splitted_preds)
        if wts is None:
            # wts = np.ones(length, dtype=np.float32) / length
            wts = [1.0 / length for _ in range(length)]
        else:
            wts = [float(el) for el in wts]

        if task.name == "multiclass":
            assert isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be an array or vector, but {type(role)}"

            vec_role = cast(NumericVectorOrArrayRole, role)
            wfeat_role = NumericVectorOrArrayRole(
                vec_role.size,
                f"WeightedBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=vec_role.is_vector
            )

            def treat_nans(w: float):
                return [
                    F.transform(feat, lambda x: F.when(F.isnan(x), 0.0).otherwise(x * w)).alias(feat),
                    F.when(F.array_contains(feat, float('nan')), 0.0).otherwise(w).alias(nan_feat)
                ]

            def sum_predictions(summ_sdf: SparkDataFrame, curr_sdf: SparkDataFrame):
                return F.transform(F.arrays_zip(summ_sdf[feat], curr_sdf[feat]), lambda x, y: x + y).alias(feat)

            normalize_weighted_sum_col = (
                F.when(F.col(nan_feat) == 0.0, None)
                .otherwise(F.transform(feat, lambda x: x / F.col(nan_feat)))
                .alias(wfeat_name)
            )
        else:
            assert isinstance(role, NumericRole) and not isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be numeric, but {type(role)}"

            wfeat_role = NumericRole(np.float32, prob=self._outp_prob)

            def treat_nans(w):
                return [
                    (F.col(feat) * w).alias(feat),
                    F.when(F.isnan(feat), 0.0).otherwise(w).alias(nan_feat)
                ]

            def sum_predictions(summ_sdf: SparkDataFrame, curr_sdf: SparkDataFrame):
                return (summ_sdf[feat] + curr_sdf[feat]).alias(feat)

            normalize_weighted_sum_col = (
                F.when(F.col(nan_feat) == 0.0, float('nan'))
                .otherwise(F.col(feat) / F.col(nan_feat))
                .alias(wfeat_name)
            )

        sum_with_nans_sdf = [
            x.data.select(
                SparkDataset.ID_COLUMN,
                *treat_nans(w)
            )
            for (x, w) in zip(splitted_preds, wts)
        ]

        sum_sdf = sum_with_nans_sdf[0]
        for sdf in sum_with_nans_sdf[1:]:
            sum_sdf = (
                sum_sdf
                .join(sdf, on=SparkDataset.ID_COLUMN)
                .select(
                    sum_sdf[SparkDataset.ID_COLUMN],
                    sum_predictions(sum_sdf, sdf),
                    (sum_sdf[nan_feat] + sdf[nan_feat]).alias(nan_feat)
                )
            )

        # TODO: SPARK-LAMA potentially this is a bad place check it later:
        #  1. equality condition double types
        #  2. None instead of nan (in the origin)
        #  due to Spark doesn't allow to mix types in the same column
        weighted_sdf = sum_sdf.select(
            SparkDataset.ID_COLUMN,
            normalize_weighted_sum_col
        )

        output = splitted_preds[0].empty()
        output.set_data(weighted_sdf, [wfeat_name], wfeat_role)

        return output


class SparkMeanBlender(SparkBlender):

    def _fit_predict(self,
                     predictions: SparkDataset,
                     pipes: Sequence[SparkMLPipeline]
                     ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        
        transformer = MeanBlenderTransformer(task_name=predictions.task.name, role=)
        self._transformer = transformer

        df = transformer.transform(predictions.data)

        new_roles = deepcopy(predictions.roles)
        new_roles.update(transformer.getOutputRoles())

        output = predictions.empty()
        output.set_data(data=df,
                        features=predictions.features + transformer.getOutputCols(),
                        roles=new_roles)

        return (output, pipes)


class MeanBlenderTransformer(Transformer, HasInputCols, HasOutputCols):

    _name = "MeanBlend"

    def __init__(self, input_cols: List[str], task_name: str, role: ColumnRole) -> None:
        super().__init__()
        
        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, [self._name])

        self._task_name = task_name


        if self._task_name == "multiclass":
            assert isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be an array or vector, but {type(role)}"

            vec_role = cast(NumericVectorOrArrayRole, role)
            output_role = NumericVectorOrArrayRole(
                vec_role.size,
                f"MeanBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=vec_role.is_vector
            )
            self.set(self.outputRoles, {self._name: output_role})
        else:
            assert isinstance(role, NumericRole) and not isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be numeric, but {type(role)}"

            output_role = NumericRole(np.float32, prob=self._outp_prob)

            self.set(self.outputRoles, {self._name: output_role})   

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:

        prediction_columns = self.getInputCols()
        length = len(prediction_columns)


        if self._task_name == "multiclass":

            is_nan_columns = []
            for c in prediction_columns:
                is_nan_columns.append( (F.when(F.array_contains(c, float('nan')), 1.0).otherwise(0.0)) )
            
            mean_col = F.transform(F.aggregate(F.array(prediction_columns), F.lit(0), lambda acc, x: acc + x), lambda x: x/F.lit(length))
            df = df.select('*', F.when(F.sum(*is_nan_columns) != F.lit(0), None).otherwise(mean_col).alias(self._name))

        else:
            is_nan_columns = []
            for c in prediction_columns:
                is_nan_columns.append( (F.when(F.isnan(c), 1.0).otherwise(0.0)) )
            df = df.select('*', F.when(F.sum(*is_nan_columns) != F.lit(0), None).otherwise(F.sum(*prediction_columns)/F.lit(length)).alias(self._name))

        return df 
