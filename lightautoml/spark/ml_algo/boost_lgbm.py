import logging
from copy import copy, deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from pandas import Series
from pyspark.sql import functions as F, Column
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Transformer, Estimator, PipelineModel, Pipeline
from pyspark.ml.util import MLReadable, MLWritable, MLWriter
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, HasOutputCol
from pyspark.ml.param.shared import Param, Params
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor
from lightautoml.dataset.roles import ColumnRole, NumericRole

from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, TabularMLAlgoTransformer, AveragingTransformer
from lightautoml.spark.mlwriters import TmpСommonMLWriter
# from lightautoml.spark.validation.base import TmpIterator, TrainValidIterator
import pandas as pd

from lightautoml.spark.tasks.base import Task
from lightautoml.utils.timer import TaskTimer
from lightautoml.utils.tmp_utils import log_data
from lightautoml.validation.base import TrainValidIterator

from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# LightGBM = Union[LightGBMClassifier, LightGBMRegressor]


class BoostLGBM(SparkTabularMLAlgo, ImportanceEstimator):

    _name: str = "LightGBM"

    _default_params = {
        "learningRate": 0.05,
        "numLeaves": 128,
        "featureFraction": 0.7,
        "baggingFraction": 0.7,
        "baggingFreq": 1,
        "maxDepth": -1,
        "minGainToSplit": 0.0,
        "maxBin": 255,
        "minDataInLeaf": 3,
        # e.g. num trees
        "numIterations": 3000,
        "earlyStoppingRound": 100,
        # for regression
        "alpha": 1.0,
        "lambdaL1": 0.0,
        "lambdaL2": 0.0,
        # seeds
        # "baggingSeed": 42
    }

    def __init__(self,
                 task: Task,
                 input_features: Optional[List[str]] = None,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {}):
        SparkTabularMLAlgo.__init__(self, task, input_features, default_params, freeze_defaults, timer, optimization_search_space)
        self._prediction_col = f"prediction_{self._name}"
        self._assembler = None

    def _infer_params(self) -> Tuple[dict, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        assert self.task is not None

        task = self.task.name

        params = copy(self.params)

        if "isUnbalance" in params:
            params["isUnbalance"] = True if params["isUnbalance"] == 1 else False

        verbose_eval = 1

        # TODO: SPARK-LAMA fix metrics and objectives
        # TODO: SPARK-LAMA add multiclass processing
        if task == "reg":
            params["objective"] = "regression"
            params["metric"] = "mse"
        elif task == "binary":
            params["objective"] = "binary"
            params["metric"] = "auc"
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
        else:
            raise ValueError(f"Unsupported task type: {task}")

        # get objective params
        # TODO SPARK-LAMA: Only for smoke test
        loss = None  # self.task.losses["lgb"]
        # params["objective"] = None  # loss.fobj_name
        fobj = None  # loss.fobj

        # get metric params
        # params["metric"] = None  # loss.metric_name
        feval = None  # loss.feval

        # params["num_class"] = None  # self.n_classes
        # add loss and tasks params if defined
        # params = {**params, **loss.fobj_params, **loss.metric_params}

        if task != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        params = {**params}

        return params, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        # TODO: SPARK-LAMA doing it to make _get_default_search_spaces working
        self.task = train_valid_iterator.train.task

        sds = cast(SparkDataset, train_valid_iterator.train)
        # TODO: SPARK-LAMA may be expensive
        rows_num = sds.data.count()
        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == "reg":
            suggested_params = {
                "learningRate": 0.05,
                "numLeaves": 32,
                "featureFraction": 0.9,
                "baggingFraction": 0.9,
            }

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 1200
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params["numLeaves"] = 128 if task == "reg" else 244
        elif rows_num > 100000:
            suggested_params["numLeaves"] = 64 if task == "reg" else 128
        elif rows_num > 50000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.0
        elif rows_num > 10000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.2
        elif rows_num > 5000:
            suggested_params["numLeaves"] = 24 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.5
        else:
            suggested_params["numLeaves"] = 16 if task == "reg" else 16
            suggested_params["alpha"] = 1 if task == "reg" else 1

        suggested_params["learningRate"] = init_lr
        suggested_params["numIterations"] = ntrees
        suggested_params["earlyStoppingRound"] = es

        if task != "reg":
            if "alpha" in suggested_params:
                del suggested_params["alpha"]

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
        assert self.task is not None

        optimization_search_space = dict()

        optimization_search_space["featureFraction"] = SearchSpace(
            Distribution.UNIFORM,
            low=0.5,
            high=1.0,
        )

        optimization_search_space["numLeaves"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=16,
            high=255,
        )

        if self.task.name == "binary" or self.task.name == "multiclass":
            optimization_search_space["isUnbalance"] = SearchSpace(
                Distribution.DISCRETEUNIFORM,
                low=0,
                high=1,
                q=1
            )

        if estimated_n_trials > 30:
            optimization_search_space["baggingFraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            # # TODO: SPARK-LAMA is there an alternative in synapse ml ?
            # optimization_search_space["min_sum_hessian_in_leaf"] = SearchSpace(
            #     Distribution.LOGUNIFORM,
            #     low=1e-3,
            #     high=10.0,
            # )

        if estimated_n_trials > 100:
            if self.task.name == "reg":
                optimization_search_space["alpha"] = SearchSpace(
                    Distribution.LOGUNIFORM,
                    low=1e-8,
                    high=10.0,
                )

            optimization_search_space["lambdaL1"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

        return optimization_search_space

    def predict_single_fold(self,
                            dataset: SparkDataset,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        temp_sdf = self._assembler.transform(dataset.data)

        pred = model.transform(temp_sdf)

        return pred

    def fit_predict_single_fold(self, fold_prediction_column: str, train: SparkDataset, valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        assert self.validation_column in train.data.columns, 'Train should contain validation column'

        if self.task is None:
            self.task = train.task

        (
            params,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        logger.info(f"Input cols for the vector assembler: {train.features}")
        logger.info(f"Running lgb with the following params: {params}")

        # TODO: reconsider using of 'keep' as a handleInvalid value
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=self._input_features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        LGBMBooster = LightGBMRegressor if train.task.name == "reg" else LightGBMClassifier

        if train.task.name == "multiclass":
            params["probabilityCol"] = fold_prediction_column

        lgbm = LGBMBooster(
            **params,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=train.target_column,
            predictionCol=fold_prediction_column if train.task.name != "multiclass" else "prediction",
            validationIndicatorCol=self.validation_column,
            verbosity=verbose_eval
        )

        logger.info(f"In GBM with params: {lgbm.params}")

        if train.task.name == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        temp_sdf = self._assembler.transform(train.data)

        ml_model = lgbm.fit(temp_sdf)

        val_pred = ml_model.transform(self._assembler.transform(valid.data))
        val_pred = val_pred.select('*', fold_prediction_column)

        return ml_model, val_pred, fold_prediction_column

    def fit(self, train_valid: TrainValidIterator):
        self.fit_predict(train_valid)

    def get_features_score(self) -> Series:
        imp = 0
        for model in self.models:
            imp = imp + pd.Series(model.getFeatureImportances(importance_type='gain'))

        imp = imp / len(self.models)

        result = Series(list(imp), index=self.features).sort_values(ascending=False)
        return result

    def _build_transformer(self) -> Transformer:
        avr = AveragingTransformer(self.task.name, input_cols=self._models_prediction_columns, output_col=self.prediction_feature)
        averaging_model = PipelineModel(stages=[self._assembler] + self.models + [avr])
        return averaging_model



# outputRoles = Param(Params._dummy(), "outputRoles",
#                         "output roles (lama format)")

# taskName = Param(Params._dummy(), "taskName",
#                         "Task type name: 'req', 'binary' or 'multiclass'")

class BoostLGBMTransformer(TabularMLAlgoTransformer, MLWritable):
    """BoostLGBM Spark MLlib Transformer"""

    def __init__(self, assembler, models, predict_feature_name, task_name, n_classes):
        super().__init__()
        self._models = models
        self._assembler = assembler
        self._predict_feature_name = predict_feature_name
        self._task_name = task_name
        self._n_classes = n_classes

    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return TmpСommonMLWriter(self.uid)

    def predict_single_fold(self,
                            dataset: SparkDataFrame,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        temp_sdf = self._assembler.transform(dataset)

        pred = model.transform(temp_sdf)

        return pred
