"""Base classes for machine learning algorithms."""

import logging
import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from pyspark.sql import functions as F

from lightautoml.dataset.roles import NumericRole
from lightautoml.utils.timer import PipelineTimer, TaskTimer

from lightautoml.spark.validation.base import TrainValidIterator
from lightautoml.spark.dataset.base import SparkDataset


logger = logging.getLogger(__name__)
SparkTabularDataset = SparkDataset


class SparkMLAlgo(ABC):
    """
    Abstract class for machine learning algorithm.
    Assume that features are already selected,
    but parameters my be tuned and set before training.
    """

    _default_params: Dict = {}
    optimization_search_space: Dict = {}
    # TODO: add checks here
    _fit_checks: Tuple = ()
    _transform_checks: Tuple = ()
    _params: Dict = None
    _name = "AbstractAlgo"

    @property
    def name(self) -> str:
        """Get model name."""
        return self._name

    @property
    def features(self) -> List[str]:
        """Get list of features."""
        return self._features

    @features.setter
    def features(self, val: Sequence[str]):
        """List of features."""
        self._features = list(val)

    @property
    def is_fitted(self) -> bool:
        """Get flag is the model fitted or not."""
        return self.features is not None

    @property
    def params(self) -> dict:
        """Get model's params dict."""
        if self._params is None:
            self._params = copy(self.default_params)
        return self._params

    @params.setter
    def params(self, new_params: dict):
        assert isinstance(new_params, dict)
        self._params = {**self.params, **new_params}

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        return self.params

    # TODO: Think about typing
    def __init__(self,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {}):

        self.task = None
        self.optimization_search_space = optimization_search_space

        self.freeze_defaults = freeze_defaults
        if default_params is None:
            default_params = {}

        self.default_params = {**self._default_params, **default_params}

        self.models = []
        self._features = None

        self.timer = timer
        if timer is None:
            self.timer = PipelineTimer().start().get_task_timer()

        self._nan_rate = None

    @abstractmethod
    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> SparkDataset:
        """Abstract method.

        Fit new algo on iterated datasets and predict on valid parts.

        Args:
            train_valid_iterator: Classic cv-iterator.

        """
        # self._features = train_valid_iterator.features

    @abstractmethod
    def predict(self, test: SparkDataset) -> SparkDataset:
        """Predict target for input data.

        Args:
            test: Dataset on test.

        Returns:
            Dataset with predicted values.

        """

    def score(self, dataset: SparkDataset) -> float:
        """Score prediction on dataset with defined metric.

        Args:
            dataset: Dataset with ground truth and predictions.

        Returns:
            Metric value.

        """
        assert self.task is not None, "No metric defined. Should be fitted on dataset first."
        metric = self.task.get_dataset_metric()

        return metric(dataset, dropna=True)

    def set_prefix(self, prefix: str):
        """Set prefix to separate models from different levels/pipelines.

        Args:
            prefix: String with prefix.

        """
        self._name = "_".join([prefix, self._name])

    def set_timer(self, timer: TaskTimer) -> "SparkMLAlgo":
        """Set timer."""
        self.timer = timer

        return self


class SparkTabularMLAlgo(SparkMLAlgo):
    """Machine learning algorithms that accepts numpy arrays as input."""

    _name: str = "TabularAlgo"

    # TODO SPARK-LAMA: Probably we can pass one Spark dataset and column name with predictions?
    # TODO SPARK-LAMA: Do we really need this method? In case of SparkDataset it do nothing
    def _set_prediction(self, dataset: SparkDataset, preds_arr: SparkDataset) -> SparkDataset:
        """Insert predictions to dataset with. Inplace transformation.

        Args:
            dataset: Dataset to transform.
            preds_arr: Array with predicted values.

        Returns:
            Transformed dataset.

        """

        prefix = "{0}_prediction".format(self._name)
        prob = self.task.name in ["binary", "multiclass"]

        # TODO SPARK-LAMA: Possible join? I hope it can be done bu .withColumn during the prediction
        dataset.set_data(preds_arr.data, prefix, NumericRole(np.float32, force_input=True, prob=prob))

        return dataset

    def fit_predict_single_fold(self,
                                train: SparkTabularDataset,
                                valid: SparkTabularDataset) -> Tuple[Any, SparkDataset]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        raise NotImplementedError

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> SparkDataset:

        self.timer.start()

        assert self.is_fitted is False, "Algo is already fitted"
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        iterator_len = len(train_valid_iterator)
        if iterator_len > 1:
            logger.info("Start fitting \x1b[1m{}\x1b[0m ...".format(self._name))
            logger.debug(f"Training params: {self.params}")

        # save features names
        self._features = train_valid_iterator.features
        # get metric and loss if None
        self.task = train_valid_iterator.train.task

        # TODO SPARK-LAMA: Do we need this variable?
        preds_ds = train_valid_iterator.get_validation_data().empty()

        outp_dim = 1
        if self.task.name == "multiclass":
            # TODO SPARK-LAMA: Think about avoiding this agg (probably metadata?)
            # TODO SPARK-LAMA: Provide target column name
            target_column = "target"
            outp_dim = preds_ds.agg(F.max(F.col(target_column).cast("long"))).collect()[0][0] + 1

        self.n_classes = outp_dim

        preds_arr = np.zeros((preds_ds.shape[0], outp_dim), dtype=np.float32)
        counter_arr = np.zeros((preds_ds.shape[0], 1), dtype=np.float32)

        for n, (idx, train, valid) in enumerate(train_valid_iterator):
            if iterator_len > 1:
                logger.info2(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m =====".format(n, self._name)
                )
            self.timer.set_control_point()

            model, pred = self.fit_predict_single_fold(train, valid)
            self.models.append(model)
            preds_arr[idx] += pred.reshape((pred.shape[0], -1))
            counter_arr[idx] += 1

            self.timer.write_run_info()

            if (n + 1) != len(train_valid_iterator):
                # split into separate cases because timeout checking affects parent pipeline timer
                if self.timer.time_limit_exceeded():
                    logger.info("Time limit exceeded after calculating fold {0}\n".format(n))
                    break

        preds_arr /= np.where(counter_arr == 0, 1, counter_arr)
        preds_arr = np.where(counter_arr == 0, np.nan, preds_arr)

        preds_ds = self._set_prediction(preds_ds, preds_arr)

        if iterator_len > 1:
            logger.info(f"Fitting \x1b[1m{self._name}\x1b[0m finished. score = \x1b[1m{self.score(preds_ds)}\x1b[0m")

        if iterator_len > 1 or "Tuned" not in self._name:
            logger.info("\x1b[1m{}\x1b[0m fitting and predicting completed".format(self._name))
        return preds_ds

    def predict_single_fold(self, model: Any, dataset: SparkTabularDataset) -> SparkDataset:

        raise NotImplementedError

    def predict(self, dataset: SparkTabularDataset) -> SparkDataset:

        assert self.models != [], "Should be fitted first."
        preds_ds = dataset.empty().to_numpy()
        preds_arr = None

        for model in self.models:
            if preds_arr is None:
                preds_arr = self.predict_single_fold(model, dataset)
            else:
                preds_arr += self.predict_single_fold(model, dataset)

        preds_arr /= len(self.models)
        preds_arr = preds_arr.reshape((preds_arr.shape[0], -1))
        preds_ds = self._set_prediction(preds_ds, preds_arr)

        return preds_ds
