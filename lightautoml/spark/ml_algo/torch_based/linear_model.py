import sys
from typing import Sequence, Optional, Callable

import torch
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from torch import optim

from lightautoml.ml_algo.torch_based.linear_model import ArrayOrSparseMatrix
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame

import horovod.spark.torch as hvd


class SparkTorchBasedLinearEstimator:
    """Linear model based on torch L-BFGS solver.

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """

    def __init__(
        self,
        data_size: int,
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
        metric=Optional[Callable],
    ):
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
        self.data_size = data_size
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

    def _prepare_data_dense(self, data: np.ndarray):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if 0 < len(self.categorical_idx) < data.shape[1]:
            data_cat = torch.from_numpy(data[:, self.categorical_idx].astype(np.int64))
            data = torch.from_numpy(data[:, np.setdiff1d(np.arange(data.shape[1]), self.categorical_idx)])
            return data, data_cat

        elif len(self.categorical_idx) == 0:
            data = torch.from_numpy(data)
            return data, None

        else:
            data_cat = torch.from_numpy(data.astype(np.int64))
            return None, data_cat

    def _optimize(
        self,
        train_ds: SparkDataset,
        val_ds: SparkDataset,
        weights_col: Optional[str] = None,
        c: float = 1
    ):
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        # self.model.train()
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
        torch_estimator = hvd.TorchEstimator(
            backend=backend,
            store=store,
            model=self.model,
            optimizer=opt,
            train_minibatch_fn=_train_minibatch_fn(),
            loss=lambda input, target: self._loss_fn(input, target.long(), None, c),
            input_shapes=[[-1, 1, 28, 28]],
            feature_cols=['features'],
            label_cols=[train_ds.target_column],
            batch_size=128,
            epochs=self.max_iter,
            validation=0.1,
            verbose=1
        )

        torch_model = torch_estimator.fit(train_ds.data).setOutputCols(['label_prob'])
        pred_df = torch_model.transform(val_ds.data)

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

    def fit(
        self,
        ds: SparkDataset,
        weights_col: Optional[str] = None
    ):
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
        data, data_cat = self._prepare_data(data)
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        y = torch.from_numpy(y.astype(np.float32))
        if weights is not None:
            weights = torch.from_numpy(weights.astype(np.float32))

        if data_val is None and y_val is None:
            logger.info2("Validation data should be defined. No validation will be performed and C = 1 will be used")
            self._optimize(data, data_cat, y, weights, 1.0)

            return self

        data_val, data_val_cat = self._prepare_data(data_val)

        best_score = -np.inf
        best_model = None
        es = 0

        for c in self.cs:
            self._optimize(data, data_cat, y, weights, c)

            val_pred = self._score(data_val, data_val_cat)
            score = self.metric(y_val, val_pred, weights_val)
            logger.info3("Linear model: C = {0} score = {1}".format(c, score))
            if score > best_score:
                best_score = score
                best_model = deepcopy(self.model)
                es = 0
            else:
                es += 1

            if es >= self.early_stopping:
                break

        self.model = best_model

        return self

    def _score(self, data: np.ndarray, data_cat: Optional[np.ndarray]) -> np.ndarray:
        """Get predicts to evaluate performance of model.

        Args:
            data: Numeric data.
            data_cat: Categorical data.

        Returns:
            Predicted target values.

        """
        with torch.set_grad_enabled(False):
            self.model.eval()
            preds = self.model(data, data_cat).numpy()

        return preds

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Inference phase.

        Args:
            data: Data to test.

        Returns:
            Predicted target values.

        """
        data, data_cat = self._prepare_data(data)

        return self._score(data, data_cat)