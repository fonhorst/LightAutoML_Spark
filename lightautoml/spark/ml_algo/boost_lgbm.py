import logging
from copy import copy
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
from pandas import Series
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.ml_algo.base import TabularMLAlgo, SparkMLModel
from lightautoml.utils.tmp_utils import log_data
from lightautoml.validation.base import TrainValidIterator

from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# LightGBM = Union[LightGBMClassifier, LightGBMRegressor]


class BoostLGBM(TabularMLAlgo, ImportanceEstimator):

    _name: str = "LightGBM"

    _default_params = {
        "task": "train",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 1,
        "reg_lambda": 0.0,
        "min_split_gain": 0.0,
        "zero_as_missing": False,
        "num_threads": 4,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "num_trees": 3000,
        "early_stopping_rounds": 100,
        "random_state": 42,
    }

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self._prediction_col = f"prediction_{self._name}"
        self.params = {} if params is None else params
        self.task = None

        self._features_importance = None
        self._assembler = None

    def _infer_params(self) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        # TODO: Check how it works with custom tasks
        params = copy(self.params)
        early_stopping_rounds = params.pop("early_stopping_rounds")
        num_trees = params.pop("num_trees")

        verbose_eval = True

        # get objective params
        # TODO SPARK-LAMA: Only for smoke test
        loss = None  # self.task.losses["lgb"]
        params["objective"] = None  # loss.fobj_name
        fobj = None  # loss.fobj

        # get metric params
        params["metric"] = None  # loss.metric_name
        feval = None  # loss.feval

        params["num_class"] = None  # self.n_classes
        # add loss and tasks params if defined
        # params = {**params, **loss.fobj_params, **loss.metric_params}
        params = {**params}

        return params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        # TODO SPARK-LAMA: Only for smoke test
        try:
            is_reg = train_valid_iterator.train.task.name == "reg"
        except AttributeError:
            is_reg = False

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if is_reg:
            suggested_params = {
                "learning_rate": 0.05,
                "num_leaves": 32,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
            }

        suggested_params["num_leaves"] = 128 if is_reg else 244

        suggested_params["learning_rate"] = 0.05
        suggested_params["num_trees"] = 2000
        suggested_params["early_stopping_rounds"] = 100

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        optimization_search_space = {}

        optimization_search_space["feature_fraction"] = SearchSpace(
            Distribution.UNIFORM,
            low=0.5,
            high=1.0,
        )

        optimization_search_space["num_leaves"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=16,
            high=255,
        )

        if estimated_n_trials > 30:
            optimization_search_space["bagging_fraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            optimization_search_space["min_sum_hessian_in_leaf"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            optimization_search_space["reg_alpha"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )
            optimization_search_space["reg_lambda"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

        return optimization_search_space

    def predict_single_fold(self,
                            dataset: SparkDataset,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        log_data("spark_lgb_predict", {"predict": dataset.to_pandas()})

        # assembler = VectorAssembler(
        #     inputCols=dataset.features,
        #     outputCol=f"{self._name}_vassembler_features",
        #     handleInvalid="keep"
        # )

        temp_sdf = self._assembler.transform(dataset.data)

        pred = model.transform(temp_sdf)

        return pred

    def fit_predict_single_fold(self, train: SparkDataset, valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        if self.task is None:
            self.task = train.task

        is_reg = self.task.name == "reg"

        (
            params,
            num_trees,
            early_stopping_rounds,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        log_data("spark_lgb_train_val", {"train": train.to_pandas(), "valid": valid.to_pandas()})

        is_val_col = 'is_val'

        train_sdf = self._make_sdf_with_target(train).withColumn(is_val_col, F.lit(0))
        valid_sdf = self._make_sdf_with_target(valid).withColumn(is_val_col, F.lit(1))

        train_valid_sdf = train_sdf.union(valid_sdf)

        # from pyspark.sql import functions as F
        # dump_sdf = train_sdf.select([F.col(c).alias(c.replace('(', '___').replace(')', '___')) for c in train.data.columns])
        # # dump_sdf.coalesce(1).write.parquet("file:///spark_data/tmp_selector_lgbm_0125l.parquet", mode="overwrite")
        # dump_pdf = dump_sdf.toPandas()#.write.parquet("file:///spark_data/tmp_selector_lgbm_0125l.parquet", mode="overwrite")
        #
        # import pickle
        # with open("/spark_data/dump_selector_lgbm_0125l.pickle", "wb") as f:
        #     pickle.dump(dump_pdf, f)

        logger.info(f"Input cols for the vector assembler: {train.features}")
        # TODO: reconsider using of 'keep' as a handleInvalid value
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=train.features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        LGBMBooster = LightGBMRegressor if is_reg else LightGBMClassifier
        metric = 'mse' if is_reg else 'auc'

        lgbm = LGBMBooster(
            isUnbalance=True,
            # baggingSeed=42,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=train.target_column,
            predictionCol=self._prediction_col,
            validationIndicatorCol=is_val_col,
            objective='binary',
            learningRate=0.02,
            numLeaves=10,
            featureFraction=0.7,
            baggingFraction=0.7,
            baggingFreq=1,
            maxDepth=-1,
            verbosity=-1,
            minGainToSplit=0.0,
            numThreads=1,
            maxBin=255,
            minDataInLeaf=3,
            earlyStoppingRound=100,
            metric=metric,
            numIterations=3000
            # numIterations=1
        )

        logger.info(f"In GBM with params: {lgbm.params}")

        if is_reg:
            # lgbm.setAlpha(params["reg_alpha"]).setLambdaL1(params["reg_lambda"]).setLambdaL2(params["reg_lambda"])
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        temp_sdf = self._assembler.transform(train_valid_sdf)

        ml_model = lgbm.fit(temp_sdf)

        val_pred = ml_model.transform(self._assembler.transform(valid_sdf))

        # TODO: dummy feature importance, need to be replaced
        self._features_importance = pd.Series(
            [1.0 / len(train.features) for _ in train.features],
            index=list(train.features)
        )

        return ml_model, val_pred, self._prediction_col

    def fit(self, train_valid: TrainValidIterator):
        self.fit_predict(train_valid)

    def get_features_score(self) -> Series:
        return self._features_importance
