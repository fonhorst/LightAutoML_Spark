import logging
import sys
from typing import Sequence, Optional, Callable, Dict, List, Any, Union

import horovod.spark.torch as hvd
import numpy as np
import pyspark.sql.functions as F
import torch
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import HasPredictionCol
from torch import nn
from torch import optim

from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.ml_algo.torch_based.linear_model import CatRegression, CatLogisticRegression, CatMulticlass
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.tasks.base import DEFAULT_PREDICTION_COL_NAME, DEFAULT_TARGET_COL_NAME
from lightautoml.spark.transformers.base import SparkBaseEstimator, DropColumnsTransformer
from lightautoml.tasks.losses import TorchLossWrapper

logger = logging.getLogger(__name__)


class SparkTorchBasedLinearEstimator(SparkBaseEstimator, HasPredictionCol):
    def __init__(self,
                 input_roles: Dict[str, ColumnRole],
                 label_col: str,
                 prediction_col: str,
                 prediction_role: ColumnRole,
                 embed_sizes: Dict[str, int],
                 val_df: Optional[SparkDataFrame] = None,
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
                 loss: Optional[Callable] = None,
                 metric: Optional[Callable] = None):
        """
        Args:
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
        self.embed_sizes = embed_sizes
        self.val_df = val_df
        self.set(self.predictionCol, prediction_col)

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

        Returns:
            self.

        """
        assert self.model is not None, "Model should be defined"

        if self._weights_col is not None:
            n = data.select(F.sum(self._weights_col)).first()
        else:
            n = data.count()

        if not self.val_df:
            logger.info("Validation data should be defined. No validation will be performed and C = 1 will be used")
            return self._optimize(data, n, 1.0)

        best_score = -np.inf
        best_model = None
        es = 0

        for c in self.cs:
            model = self._optimize(data, n, c)
            val_pred = (
                model
                .transform(self.val_df)
            )

            val_pred = (
                val_pred
                .select(
                    SparkDataset.ID_COLUMN,
                    F.col(self.label_col).alias(DEFAULT_TARGET_COL_NAME),
                    F.col(self.getPredictionCol()).alias(DEFAULT_PREDICTION_COL_NAME)
                )
            )
            score = self.metric(val_pred)
            logger.info(f"Spark TorchBased Linear model: C = {c} score = {score}")
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

    def _optimize(self, train_df: SparkDataFrame, n: Union[int, float], c: float = 1) -> Transformer:
        """Optimize single model.

        Args:
            train_df: Dataframe with numerical and categorical data
                (two different columns containing vectors) to train.
            n: either size of data or sum of weights
            c: Regularization coefficient.

        """
        numeric_feats = self._get_numeric_feats()
        cat_feats = self._get_cat_feats()

        numeric_assembler = VectorAssembler(inputCols=numeric_feats, outputCol="numeric_features")
        cat_assembler = VectorAssembler(inputCols=cat_feats, outputCol="cat_features")

        opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.5)

        _loss_fn = self._loss_fn

        def _train_minibatch_fn():
            def train_minibatch(model, optimizer, transform_outputs, loss_fn, inputs, labels, sample_weights):
                optimizer.zero_grad()
                outputs = model(*inputs)
                outputs, labels = transform_outputs(outputs, labels)
                loss = _loss_fn(loss_fn, model, labels, outputs, sample_weights, c, n)
                loss.backward()
                optimizer.step()
                return outputs, loss
            return train_minibatch

        # Setup our store for intermediate data
        store = Store.create('/tmp/hvd_spark')

        backend = SparkBackend(
            num_proc=1,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        torch_estimator = hvd.TorchEstimator(
            backend=backend,
            store=store,
            model=self.model,
            optimizer=opt,
            train_minibatch_fn=_train_minibatch_fn(),
            loss=self.loss,
            input_shapes=[[-1, len(numeric_feats)], [-1, len(cat_feats)]],
            feature_cols=[numeric_assembler.getOutputCol(), cat_assembler.getOutputCol()],
            label_cols=[self.label_col],
            batch_size=1024,
            epochs=self.max_iter,
            verbose=2
        )
        drop_columns = DropColumnsTransformer(remove_cols=torch_estimator.getFeatureCols())

        if self.output_size > 1:
            cols = [c for c in train_df.columns if c != self.label_col]
            train_df = train_df.select(*cols, F.col(self.label_col).astype('int').alias(self.label_col))

        sdf = numeric_assembler.transform(train_df)
        sdf = cat_assembler.transform(sdf)
        torch_model = torch_estimator.fit(sdf).setOutputCols([self.getPredictionCol()])

        return PipelineModel(stages=[numeric_assembler, cat_assembler, torch_model, drop_columns])

    def _get_numeric_feats(self) -> List[str]:
        feats = [
            feat for feat, role in self.getInputRoles().items()
            if isinstance(role, NumericRole)
        ]
        return sorted(feats)

    def _get_cat_feats(self) -> List[str]:
        feats = [
            feat for feat, role in self.getInputRoles().items()
            if isinstance(role, CategoryRole)
        ]
        return sorted(feats)

    @staticmethod
    def _loss_fn(
            loss_fn: Callable,
            model: Any,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            weights: Optional[torch.Tensor],
            c: float,
            n: Union[int, float]
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
        # raise ValueError(f"y_true: {len(y_true)}, {y_true[0].type()}, {y_true[0].shape}")

        loss = loss_fn([y_true[0].long()], y_pred, sample_weights=weights)

        all_params = torch.cat([y.view(-1) for (x, y) in model.named_parameters() if x != "bias"])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return loss + 0.5 * penalty / c


class SparkTorchBasedLinearRegression(SparkTorchBasedLinearEstimator):
    def __init__(self,
                 input_roles: Dict[str, ColumnRole],
                 label_col: str,
                 prediction_col: str,
                 prediction_role: ColumnRole,
                 embed_sizes: Dict[str, int],
                 val_df: Optional[SparkDataFrame] = None,
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
                 loss: Optional[Callable] = None,
                 metric: Optional[Callable] = None):
        """
        Args:
            embed_sizes: categorical embedding sizes
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if loss is None:
            loss = TorchLossWrapper(nn.MSELoss)

        super().__init__(input_roles, label_col, prediction_col, prediction_role, embed_sizes, val_df,
                         1, cs, max_iter, tol, early_stopping, loss, metric)

        numeric_feats = self._get_numeric_feats()
        cat_feats = self._get_cat_feats()
        embed_sizes = [self.embed_sizes[feat] for feat in cat_feats]

        self.model = CatRegression(
            len(numeric_feats),
            embed_sizes,
            self.output_size,
        )
    """Torch-based linear regressor optimized by L-BFGS."""


class SparkTorchBasedLogisticRegression(SparkTorchBasedLinearEstimator):
    """Linear binary classifier."""
    def __init__(self, input_roles: Dict[str, ColumnRole], label_col: str, prediction_col: str,
                 prediction_role: ColumnRole, embed_sizes: Dict[str, int], val_df: Optional[SparkDataFrame] = None,
                 output_size: int = 1, cs: Sequence[float] = (
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
                 max_iter: int = 1000, tol: float = 1e-5, early_stopping: int = 2,
                 loss: Optional[Callable] = None,
                 metric: Optional[Callable] = None):
        """
        Args:
            embed_sizes: categorical embedding sizes.
            output_size: size of output layer.
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if output_size == 1:
            _loss = nn.BCELoss
            _model = CatLogisticRegression
            self._binary = True
        else:
            _loss = nn.CrossEntropyLoss
            _model = CatMulticlass
            self._binary = False

        if loss is None:
            loss = TorchLossWrapper(_loss)

        super().__init__(input_roles, label_col, prediction_col, prediction_role, embed_sizes, val_df, output_size, cs,
                         max_iter, tol, early_stopping, loss, metric)

        numeric_feats = self._get_numeric_feats()
        cat_feats = self._get_cat_feats()
        embed_sizes = [self.embed_sizes[feat] for feat in cat_feats]

        self.model = _model(
            len(numeric_feats),
            embed_sizes,
            self.output_size,
        )
