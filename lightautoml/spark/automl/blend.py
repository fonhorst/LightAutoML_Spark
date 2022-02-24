from abc import ABC
from copy import copy
from typing import List, Optional, Sequence, Tuple, cast, Callable

import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, Param
from pyspark.ml.util import MLWritable
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan

from lightautoml.automl.blend import WeightedBlender
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.reader.base import RolesDict
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.base import AveragingTransformer
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.tasks.base import DEFAULT_PREDICTION_COL_NAME, SparkTask
from lightautoml.spark.transformers.base import ColumnsSelectorTransformer


class SparkBlender(ABC):
    """Basic class for blending.

    Blender learns how to make blend
    on sequence of prediction datasets and prune pipes,
    that are not used in final blend.

    """

    def __init__(self):
        self._transformer = None
        self._single_prediction_col_name = DEFAULT_PREDICTION_COL_NAME
        self._pred_role: Optional[ColumnRole] = None
        self._output_roles: Optional[RolesDict] = None
        self._task: Optional[SparkTask] = None

    @property
    def output_roles(self) -> RolesDict:
        assert self._output_roles is not None, "Blender has not been fitted yet"
        return self._output_roles

    @property
    def transformer(self) -> Transformer:
        """Returns Spark MLlib Transformer.
        Represents a Transformer with fitted models."""

        assert self._transformer is not None, "Pipeline is not fitted!"

        return self._transformer

    def fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            self._transformer = ColumnsSelectorTransformer(
                input_cols=[SparkDataset.ID_COLUMN] + list(pipes[0].output_roles.keys()),
                optional_cols=[predictions.target_column] if predictions.target_column else []
            )
            self._output_roles = copy(predictions.roles)
            return predictions, pipes

        self._set_metadata(predictions, pipes)

        return self._fit_predict(predictions, pipes)

    def predict(self, predictions: SparkDataset) -> SparkDataset:
        sdf = self._transformer.transform(predictions.data)

        sds = predictions.empty()
        sds.set_data(sdf, list(self.output_roles.keys()), self.output_roles)

        return sds

    def _fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
        -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        raise NotImplementedError()

    def split_models(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
            -> List[Tuple[str, int, int]]:
        """Split predictions by single model prediction datasets.

        Args:
            predictions: Dataset with predictions.

        Returns:
            Each tuple in the list is:
            - prediction column name
            - corresponding model index (in the pipe)
            - corresponding pipe index

        """
        return [
            (ml_algo.prediction_feature, j, i)
            for i, pipe in enumerate(pipes)
            for j, ml_algo in enumerate(pipe.ml_algos)
        ]

    def _set_metadata(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]):
        self._pred_role = predictions.roles[pipes[0].ml_algos[0].prediction_feature]
        self._output_roles = {self._single_prediction_col_name: self._pred_role}

        if isinstance(self._pred_role, NumericVectorOrArrayRole):
            self._outp_dim = self._pred_role.size
        else:
            self._outp_dim = 1
        self._outp_prob = predictions.task.name in ["binary", "multiclass"]
        self._score = predictions.task.get_dataset_metric()
        self._task = predictions.task

    def _make_single_pred_ds(self, predictions: SparkDataset, pred_col: str) -> SparkDataset:
        pred_sdf = predictions.data.select(
            SparkDataset.ID_COLUMN,
            predictions.target_column,
            F.col(pred_col).alias(self._single_prediction_col_name)
        )
        pred_roles = {c: predictions.roles[c] for c in pred_sdf.columns}
        pred_ds = predictions.empty()
        pred_ds.set_data(pred_sdf, pred_sdf.columns, pred_roles)

        return pred_ds

    def score(self, dataset: SparkDataset) -> float:
        """Score metric for blender.

        Args:
            dataset: Blended predictions dataset.

        Returns:
            Metric value.

        """
        return self._score(dataset, True)


class SparkBestModelSelector(SparkBlender, WeightedBlender):
    def _fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
            -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        """Simple fit - just take one best.

                Args:
                    predictions: Sequence of datasets with predictions.
                    pipes: Sequence of pipelines.

                Returns:
                    Single prediction dataset and Sequence of pruned pipelines.

                """
        splitted_models_and_pipes = self.split_models(predictions, pipes)

        best_pred = None
        best_pipe_idx = 0
        best_model_idx = 0
        best_score = -np.inf

        for pred_col, mod, pipe in splitted_models_and_pipes:
            pred_ds = self._make_single_pred_ds(predictions, pred_col)
            score = self.score(pred_ds)

            if score > best_score:
                best_pipe_idx = pipe
                best_model_idx = mod
                best_score = score
                best_pred = pred_ds

        best_pipe = pipes[best_pipe_idx]
        best_pipe.ml_algos = [best_pipe.ml_algos[best_model_idx]]

        self._transformer = ColumnsSelectorTransformer(
            input_cols=[SparkDataset.ID_COLUMN, self._single_prediction_col_name]
        )

        self._output_roles = copy(best_pred.roles)

        return best_pred, [best_pipe]


class SparkWeightedBlender(SparkBlender, WeightedBlender):
    def __init__(self, max_iters: int = 5, max_inner_iters: int = 7, max_nonzero_coef: float = 0.05,):
        super().__init__()
        super(WeightedBlender, self).__init__(max_iters, max_inner_iters, max_nonzero_coef)
        self._predictions_dataset: Optional[SparkDataset] = None

    def _get_weighted_pred(self, splitted_preds: Sequence[str], wts: Optional[np.ndarray]) -> SparkDataset:
        avr = AveragingTransformer(
            task_name=self._task.name,
            input_cols=list(splitted_preds),
            output_col=self._single_prediction_col_name,
            remove_cols=list(splitted_preds),
            convert_to_array_first=True,
            weights=wts,
            dim_num=self._outp_dim
        )

        weighted_preds_sdf = avr.transform(self._predictions_dataset.data)

        wpreds_sds = self._predictions_dataset.empty()
        wpreds_sds.set_data(weighted_preds_sdf, list(self.output_roles.keys()), self.output_roles)

        return wpreds_sds

    def _fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
            -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        self._predictions_dataset = predictions

        sm = self.split_models(predictions, pipes)
        pred_cols = [pred_col for pred_col, _, _ in sm]
        pipe_idx = np.ndarray([pidx for _, _, pidx in sm])

        wts = self._optimize(pred_cols)

        reweighted_pred_cols = [x for (x, w) in zip(pred_cols, wts) if w > 0]
        pipes, self.wts = self._prune_pipe(pipes, wts, pipe_idx)
        pipes = cast(Sequence[SparkMLPipeline], pipes)

        outp = self._get_weighted_pred(reweighted_pred_cols, self.wts)

        return outp, pipes


class SparkMeanBlender(SparkBlender):
    def _fit_predict(self,
                     predictions: SparkDataset,
                     pipes: Sequence[SparkMLPipeline]
                     ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        pred_cols = [pred_col for pred_col, _, _ in self.split_models(predictions, pipes)]

        self._transformer = AveragingTransformer(
            task_name=predictions.task.name,
            input_cols=pred_cols,
            output_col=self._single_prediction_col_name,
            remove_cols=pred_cols,
            convert_to_array_first=not (predictions.task.name == "reg"),
            dim_num=self._outp_dim
        )

        df = self._transformer.transform(predictions.data)

        if predictions.task.name in ["binary", "multiclass"]:
            assert isinstance(self._pred_role, NumericVectorOrArrayRole)
            output_role = NumericVectorOrArrayRole(
                self._pred_role.size,
                f"MeanBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=self._pred_role.is_vector
            )
        else:
            output_role = NumericRole(np.float32, prob=self._outp_prob)

        roles = {f: predictions.roles[f] for f in predictions.features if f not in pred_cols}
        roles[self._single_prediction_col_name] = output_role
        pred_ds = predictions.empty()
        pred_ds.set_data(df, df.columns, roles)

        self._output_roles = copy(roles)

        return pred_ds, pipes

