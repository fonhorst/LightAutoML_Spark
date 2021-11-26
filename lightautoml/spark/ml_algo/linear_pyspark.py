"""Linear models for tabular datasets."""

import logging

from copy import copy
from typing import Tuple
from typing import Union

import numpy as np
from pyspark.ml.feature import VectorAssembler

from ..dataset.base import SparkDataset
from ..validation.base import TrainValidIterator

from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class LinearLBFGS:

    _name: str = "LinearL2"

    def __init__(self, params={}):
        self.params = params
        self.task = None

    def _infer_params(self) -> LinearEstimator:

        params = copy(self.params)
        if self.task.name in ["binary", "multiclass"]:
            model = LogisticRegression(**params)
        elif self.task.name == "reg":
            model = LinearRegression(**params)
            model.setSolver("l-bfgs")
        else:
            raise ValueError("Task not supported")

        return model

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        suggested_params = copy(self.params)
        train = train_valid_iterator.train
        suggested_params["categorical_idx"] = [
            n for (n, x) in enumerate(train.features) if train.roles[x].name == "Category"
        ]

        suggested_params["embed_sizes"] = ()
        if len(suggested_params["categorical_idx"]) > 0:
            suggested_params["embed_sizes"] = (
                train.data[:, suggested_params["categorical_idx"]].max(axis=0).astype(np.int32) + 1
            )

        suggested_params["data_size"] = train.shape[1]

        return suggested_params

    def fit_predict_single_fold(
        self, train: SparkDataset, valid: SparkDataset
    ) -> Tuple[LinearEstimatorModel, SparkDataset]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        if self.task is None:
            self.task = train.task
        model = self._infer_params()

        pr_features = train.features
        pr_features.remove(self.params["labelCol"])
        self.assembler = VectorAssembler().setInputCols(pr_features).setOutputCol(self.params["featuresCol"])

        train_assembled = self.assembler.transform(train.data)

        self.ml = model.fit(dataset=train_assembled)

        val_assembled = self.assembler.transform(valid.data)
        val_pred = self.ml.transform(val_assembled)

        return self.ml, val_pred


    def predict_single_fold(self, dataset: SparkDataset, model: LinearEstimator = None) -> SparkDataset:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        if model is None:
            model = self.ml
        data_assembled = self.assembler.transform(dataset.data)
        pred = model.transform(data_assembled)

        return pred


