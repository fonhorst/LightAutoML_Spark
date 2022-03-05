import logging
import sys
from copy import deepcopy
from typing import Sequence, Optional, Callable, List, Dict

import torch
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import HasPredictionCol
from torch import optim

from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.ml_algo.torch_based.linear_model import ArrayOrSparseMatrix
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame

import horovod.spark.torch as hvd
import numpy as np

from lightautoml.spark.transformers.base import SparkBaseEstimator
import pyspark.sql.functions as F


logger = logging.getLogger(__name__)


class SparkTorchBasedLinearEstimator(SparkBaseEstimator, HasPredictionCol):
    def __init__(self,
                 input_roles: Dict[str, ColumnRole],
                 label_col: str,
                 prediction_col: str,
                 prediction_role: ColumnRole,
                 val_col: Optional[str] = None,
                 categorical_idx: Sequence[int] = (),
                 embed_sizes: Sequence[int] = (),
                 output_size: int = 1,
                 cs: Sequence[float] = (
                    0.00001,
                    0.00005,
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                    2.0,
                    5.0,
                    7.0,
                    10.0,
                    20.0,
                 ),
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 early_stopping: int = 2,
                 loss=Optional[Callable],
                 metric=Optional[Callable]):
        """
        Args:
            data_size: Not used.
            categorical_idx: Indices of categorical features.
            embed_sizes: Categorical embedding sizes.
            output_size: Size of output layer.
            cs: Regularization coefficients.
            max_iter: Maximum iterations of L-BFGS.
            tol: Tolerance for the stopping criteria.
            early_stopping: Maximum rounds without improving.
            loss: Loss function. Format: loss(preds, true) -> loss_arr, assume ```reduction='none'```.
            metric: Metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        super().__init__(list(input_roles.keys()), input_roles, do_replace_columns=False, output_role=prediction_role)
        self.label_col = label_col
        self.val_col = val_col
        self.set(self.predictionCol, prediction_col)

        self.categorical_idx = categorical_idx
        self.embed_sizes = embed_sizes
        self.output_size = output_size

        assert all([x > 0 for x in cs]), "All Cs should be greater than 0"

        self.cs = cs
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.loss = loss  # loss(preds, true) -> loss_arr, assume reduction='none'
        self.metric = metric  # metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better)

        self._transformer: Optional[PipelineModel] = None
        self._weights_col: Optional[str] = None

    """Linear model based on torch L-BFGS solver.

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """

    def _fit(self, data: SparkDataFrame) -> Transformer:
        """Fit method.

        Args:
            data: Data to train.
            y: Train target values.
            weights: Train items weights.
            data_val: Data to validate.
            y_val: Valid target values.
            weights_val: Validation item weights.

        Returns:
            self.

        """
        assert self.model is not None, "Model should be defined"
        #
        # if len(y.shape) == 1:
        #     y = y[:, np.newaxis]
        # y = torch.from_numpy(y.astype(np.float32))
        # if weights is not None:
        #     weights = torch.from_numpy(weights.astype(np.float32))

        if not self.val_col:
            logger.info("Validation data should be defined. No validation will be performed and C = 1 will be used")
            return self._optimize(data, 1.0)

        best_score = -np.inf
        best_model = None
        es = 0

        val_df = data.where(F.col(self.val_col) == 1)

        for c in self.cs:
            model = self._optimize(data, c)
            val_pred = (
                model
                .transform(val_df)
                .select(SparkDataset.ID_COLUMN, self.label_col, self.getPredictionCol())
            )
            score = self.metric(val_pred)
            logger.info(f"Linear model: C = {c} score = {score}")
            if score > best_score:
                best_score = score
                best_model = model
                es = 0
            else:
                es += 1

            if es >= self.early_stopping:
                break

        self._transformer = best_model

        return self._transformer

    def _optimize(self, train_df: SparkDataFrame, c: float = 1) -> Transformer:
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        # self.model.train()

        numeric_feats = [feat for feat, role in self.getInputRoles().items() if isinstance(role, NumericRole)]
        cat_feats = [feat for feat, role in self.getInputRoles().items() if isinstance(role, CategoryRole)]

        numeric_assembler = VectorAssembler(inputCols=numeric_feats, outputCol="numeric_features")
        cat_assembler = VectorAssembler(inputCols=cat_feats, outputCol="cat_features")

        opt = optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=self.max_iter,
            tolerance_change=self.tol,
            tolerance_grad=self.tol,
            line_search_fn="strong_wolfe",
        )

        def _train_minibatch_fn():
            def train_minibatch(model, optimizer, transform_outputs, loss_fn, inputs, labels, sample_weights):
                optimizer.zero_grad()
                outputs = model(*inputs)
                outputs, labels = transform_outputs(outputs, labels)
                loss = loss_fn(outputs, labels, sample_weights)

                if loss.requires_grad:
                    loss.backward()

                def closure():
                    return loss
                # specific need for LBGFS optimizer
                optimizer.step(closure)
                return outputs, loss

            return train_minibatch

        # Setup our store for intermediate data
        store = Store.create('/tmp/hvd_spark')

        backend = SparkBackend(
            num_proc=1,
            stdout=sys.stdout,
            stderr=sys.stderr,
            prefix_output_with_timestamp=True
        )
        # TODO: SPARK-LAMA check for _loss_fn weights arg
        #  there should be a way to pass weights inside

        # TODO: SPARK-LAMA feature_cols = ['features', 'features_cat']
        #   we need 2 different vector assemblers to represent data
        #   as it is expected by CatLinear
        torch_estimator = hvd.TorchEstimator(
            backend=backend,
            store=store,
            model=self.model,
            optimizer=opt,
            train_minibatch_fn=_train_minibatch_fn(),
            loss=lambda input, target: self._loss_fn(input, target.long(), None, c),
            # TODO: SPARK-LAMA shapes?
            input_shapes=[[-1, 1, 28, 28]],
            feature_cols=[numeric_assembler.getOutputCol(), cat_assembler.getOutputCol()],
            label_cols=[self.label_col],
            batch_size=128,
            epochs=self.max_iter,
            # validation=0.1,
            verbose=1
        )

        sdf = numeric_assembler.transform(train_df)
        sdf = cat_assembler.transform(sdf)
        torch_model = torch_estimator.fit(sdf).setOutputCols(self.getPredictionCol())

        return PipelineModel(stages=[numeric_assembler, cat_assembler, torch_model])

    def _loss_fn(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        weights: Optional[torch.Tensor],
        c: float,
    ) -> torch.Tensor:
        """Weighted loss_fn wrapper.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            weights: Item weights.
            c: Regularization coefficients.

        Returns:
            Loss+Regularization value.

        """
        # weighted loss
        loss = self.loss(y_true, y_pred, sample_weight=weights)

        n = y_true.shape[0]
        if weights is not None:
            n = weights.sum()

        all_params = torch.cat([y.view(-1) for (x, y) in self.model.named_parameters() if x != "bias"])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return loss + 0.5 * penalty / c
