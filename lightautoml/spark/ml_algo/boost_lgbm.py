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
from pyspark.ml import Transformer, Estimator, PipelineModel
from pyspark.ml.util import MLReadable, MLWritable, MLWriter
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.param.shared import Param, Params
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor
from lightautoml.dataset.roles import ColumnRole, NumericRole

from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.base import TabularMLAlgo, SparkMLModel
from lightautoml.spark.mlwriters import TmpСommonMLWriter
# from lightautoml.spark.validation.base import TmpIterator, TrainValidIterator
import pandas as pd

from lightautoml.utils.timer import TaskTimer
from lightautoml.utils.tmp_utils import log_data
from lightautoml.validation.base import TrainValidIterator

from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# LightGBM = Union[LightGBMClassifier, LightGBMRegressor]


class BoostLGBM(TabularMLAlgo, ImportanceEstimator):

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
            default_params: Optional[dict] = None,
            freeze_defaults: bool = True,
            timer: Optional[TaskTimer] = None,
            optimization_search_space: Optional[dict] = {}):
        TabularMLAlgo.__init__(self, default_params, freeze_defaults, timer, optimization_search_space)
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

    def fit_predict_single_fold(self, train: SparkDataset, valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        if self.task is None:
            self.task = train.task

        (
            params,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        is_val_col = 'is_val'
        train_sdf = self._make_sdf_with_target(train).withColumn(is_val_col, F.lit(0))
        valid_sdf = self._make_sdf_with_target(valid).withColumn(is_val_col, F.lit(1))

        train_valid_sdf = train_sdf.union(valid_sdf)

        logger.info(f"Input cols for the vector assembler: {train.features}")
        logger.info(f"Running lgb with the following params: {params}")

        # TODO: reconsider using of 'keep' as a handleInvalid value
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=train.features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        LGBMBooster = LightGBMRegressor if train.task.name == "reg" else LightGBMClassifier

        if train.task.name == "multiclass":
            params["probabilityCol"] = self._prediction_col

        lgbm = LGBMBooster(
            **params,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=train.target_column,
            predictionCol=self._prediction_col if train.task.name != "multiclass" else "prediction",
            validationIndicatorCol=is_val_col,
            verbosity=verbose_eval
        )

        logger.info(f"In GBM with params: {lgbm.params}")

        if train.task.name == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        temp_sdf = self._assembler.transform(train_valid_sdf)

        ml_model = lgbm.fit(temp_sdf)

        val_pred = ml_model.transform(self._assembler.transform(valid_sdf))
        val_pred = val_pred.select(*valid_sdf.columns, self._prediction_col)

        return ml_model, val_pred, self._prediction_col

    def fit(self, train_valid: TrainValidIterator):
        self.fit_predict(train_valid)

    def get_features_score(self) -> Series:
        imp = 0
        for model in self.models:
            imp = imp + pd.Series(model.getFeatureImportances(importance_type='gain'))

        imp = imp / len(self.models)

        result = Series(list(imp), index=self.features).sort_values(ascending=False)
        return result


class BoostLGBMEstimator(Estimator, HasInputCols, HasOutputCols, MLWritable):
    """Spark MLlib Estimator implementation of BoostLGBM

    Uses `LightGBMClassifier` and `LightGBMRegressor` from synapse.ml.lightgbm package
    
    _fit() method return a Spark MLlib Transformer
    """

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

    inputRoles = Param(Params._dummy(), "inputRoles",
                            "input roles (lama format)")

    outputRoles = Param(Params._dummy(), "outputRoles",
                            "output roles (lama format)")

    taskName = Param(Params._dummy(), "taskName",
                            "Task type name: 'req', 'binary' or 'multiclass'")

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 task_name: Optional[str] = None,
                 folds_column: Optional[str] = None,
                 target_column: Optional[str] = None,
                 folds_number: int = 2
                 ):
        super().__init__()
        self.models = []
        self._prediction_col = f"prediction_{self._name}"
        self._task_name = task_name
        self.set(self.taskName, task_name)
        self._folds_column = folds_column
        self._target_column = target_column
        self._folds_number = folds_number
        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, [self._predict_feature_name()])
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, self.get_output_roles())
        self._is_reg = self._task_name == "reg"

    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Estimator instance."
        return TmpСommonMLWriter(self.uid)

    def get_output_roles(self):
        prob = self._task_name in ["binary", "multiclass"]

        if self._task_name == "multiclass":
            role = NumericVectorOrArrayRole(size=self.n_classes,
                                            element_col_name_template=self._predict_feature_name() + "_{}",
                                            dtype=np.float32,
                                            force_input=True,
                                            prob=prob)
        else:
            role = NumericRole(dtype=np.float32, force_input=True, prob=prob)

        new_roles = deepcopy(self.getOrDefault(self.inputRoles))
        new_roles.update({feat: role for feat in self.getOutputCols()})
        return new_roles

    def getOutputRoles(self):
        """
        Gets output roles or its default value.
        """
        return self.getOrDefault(self.outputRoles)

    def _predict_feature_name(self):
        return f"{self._name}_prediction"

    def _infer_params(self) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        # TODO: Check how it works with custom tasks
        # params = copy(self.params)
        # TODO: think over params (name taken by property from pyspark.ml.Params)
        params = copy(self.params_tmp)
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

    def init_params_on_input(self, task_name: str) -> dict:

        # suggested_params = copy(self.default_params)
        suggested_params = copy(self._default_params)

        # if self.freeze_defaults:
        #     # if user change defaults manually - keep it
        #     return suggested_params

        if self._is_reg:
            suggested_params = {
                "learning_rate": 0.05,
                "num_leaves": 32,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
            }

        suggested_params["num_leaves"] = 128 if self._is_reg else 244

        suggested_params["learning_rate"] = 0.05
        suggested_params["num_trees"] = 2000
        suggested_params["early_stopping_rounds"] = 100

        return suggested_params   

    def fit_single_fold(self, train: SparkDataFrame) -> SparkMLModel:
        # if self.task is None:
        #     self.task = train.task

        (
            params,
            num_trees,
            early_stopping_rounds,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        # train_sdf = self._make_sdf_with_target(train)
        train_sdf = train
        # valid_sdf = valid.data

        logger.info(f"Input cols for the vector assembler: {self.getInputCols()}")
        # TODO: reconsider using of 'keep' as a handleInvalid value
        assembler = VectorAssembler(
            inputCols=self.getInputCols(),
            outputCol=f"{self._name}_vassembler_features",
            handleInvalid="keep"
        )

        LGBMBooster = LightGBMRegressor if self._is_reg else LightGBMClassifier

        lgbm = LGBMBooster(
            # fobj=fobj,  # TODO SPARK-LAMA: Commented only for smoke test
            # feval=feval,
            featuresCol=assembler.getOutputCol(),
            labelCol=self._target_column,
            predictionCol=self._prediction_col,
            # learningRate=params["learning_rate"],
            # numLeaves=params["num_leaves"],
            # featureFraction=params["feature_fraction"],
            # baggingFraction=params["bagging_fraction"],
            # baggingFreq=params["bagging_freq"],
            # maxDepth=params["max_depth"],
            # verbosity=params["verbosity"],
            # minGainToSplit=params["min_split_gain"],
            # numThreads=params["num_threads"],
            # maxBin=params["max_bin"],
            # minDataInLeaf=params["min_data_in_bin"],
            # earlyStoppingRound=early_stopping_rounds
            learningRate=0.05,
            numLeaves=128,
            featureFraction=0.9,
            baggingFraction=0.9,
            baggingFreq=1,
            maxDepth=-1,
            verbosity=-1,
            minGainToSplit=0.0,
            numThreads=1,
            maxBin=255,
            minDataInLeaf=3,
            earlyStoppingRound=100,
            metric="mse",
            numIterations=2000
            # numIterations=1
        )

        logger.info(f"In GBM with params: {lgbm.params}")

        if self._is_reg:
            # lgbm.setAlpha(params["reg_alpha"]).setLambdaL1(params["reg_lambda"]).setLambdaL2(params["reg_lambda"])
            lgbm.setAlpha(1.0).setLambdaL1(0.0).setLambdaL2(0.0)

        # LGBMBooster = GBTRegressor if is_reg else GBTClassifier
        # lgbm = LGBMBooster(
        #     featuresCol=assembler.getOutputCol(),
        #     labelCol=train.target_column,
        #     predictionCol=self._prediction_col,
        #     maxDepth=5,
        #     maxBins=32,
        #     minInstancesPerNode=1,
        #     minInfoGain=0.0,
        #     cacheNodeIds=False,
        #     subsamplingRate=1.0,
        #     checkpointInterval=10,
        #     maxIter=5,
        #     impurity='variance',
        #     featureSubsetStrategy='all'
        # )

        temp_sdf = assembler.transform(train_sdf)
        ml_model = lgbm.fit(temp_sdf)

        # val_pred = ml_model.transform(assembler.transform(valid_sdf))

        # TODO: dummy feature importance, need to be replaced
        self._features_importance = pd.Series(
            [1.0 / len(self.getInputCols()) for _ in self.getInputCols()],
            index=list(self.getInputCols())
        )

        # return ml_model, val_pred, self._prediction_col
        return ml_model

    def _build_train_valid_iterator(self,
                                    dataset: SparkDataFrame, 
                                    folds_column: str,
                                    folds_number: int = 2) -> TrainValidIterator:
        raise NotImplementedError()
        # return TmpIterator(dataset, folds_column, folds_number)

    def _fit(self, dataset: SparkDataFrame) -> "BoostLGBMTransformer":
        """Divides the input dataframe into multiple parts.
        Then a separate model is trained on each part and saved.  
        
        WARNING: All this code is copied from `lightautoml.spark.ml_algo.base.TabularMLAlgo.fit_predict` to rewrite `BoostLGBM` using Spark MLlib Estimator.
        Minor edits have been made. 

        Returns:
            BoostLGBMTransformer: Transformer with fitted models
        """

        # self.timer.start()

        train_valid_iterator: TrainValidIterator = self._build_train_valid_iterator(dataset, self._folds_column, self._folds_number)

        # TODO: uncomment this if is_fitted will be added
        # assert self.is_fitted is False, "Algo is already fitted"

        # init params on input if no params was set before
        # TODO: think over self._params
        # if self._params is None:
        self.params_tmp = self.init_params_on_input(train_valid_iterator)

        iterator_len = len(train_valid_iterator)
        if iterator_len > 1:
            logger.info("Start fitting \x1b[1m{}\x1b[0m ...".format(self._name))
            logger.debug(f"Training params: {self.params_tmp}")

        # save features names
        # TODO: remove or change it
        # self._features = train_valid_iterator.features
        # get metric and loss if None
        # self.task = train_valid_iterator.train.task

        preds_ds = cast(SparkDataset, train_valid_iterator.get_validation_data())

        # spark
        outp_dim = 1
        if self._task_name == "multiclass":
            # TODO: SPARK-LAMA working with target should be reflected in SparkDataset
            tdf: SparkDataFrame = preds_ds.target
            outp_dim = tdf.select(F.max(preds_ds.target_column).alias("max")).first()
            outp_dim = outp_dim["max"] + 1

        self.n_classes = outp_dim

        # preds_arr = np.zeros((preds_ds.shape[0], outp_dim), dtype=np.float32)
        # counter_arr = np.zeros((preds_ds.shape[0], 1), dtype=np.float32)

        preds_dfs: List[SparkDataFrame] = []

        pred_col_prefix = self._predict_feature_name()

        # TODO: SPARK-LAMA - we need to cache the "parent" dataset of the train_valid_iterator
        # train_valid_iterator.cache()

        # TODO: Make parallel version later
        for n, (idx, train, valid) in enumerate(train_valid_iterator):
            if iterator_len > 1:
                logger.info2(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m =====".format(n, self._name)
                )
            # self.timer.set_control_point()

            # model, pred, prediction_column = self.fit_predict_single_fold(train, valid)
            model = self.fit_single_fold(train)
            # pred = pred.select(
            #     SparkDataset.ID_COLUMN,
            #     F.col(prediction_column).alias(f"{pred_col_prefix}_{n}")
            # )
            self.models.append(model)
            # preds_dfs.append(pred)

            # self.timer.write_run_info()

            # if (n + 1) != len(train_valid_iterator):
            #     # split into separate cases because timeout checking affects parent pipeline timer
            #     if self.timer.time_limit_exceeded():
            #         logger.info("Time limit exceeded after calculating fold {0}\n".format(n))
            #         break

        # combine predictions of all models and make them into the single one
        # full_preds_df = self._average_predictions(preds_ds, preds_dfs, pred_col_prefix)

        # TODO: send the "parent" dataset of the train_valid_iterator for unwinding later
        #       e.g. from the train_valid_iterator
        # preds_ds = self._set_prediction(preds_ds, full_preds_df)

        if iterator_len > 1:
            logger.info(
                f"Fitting \x1b[1m{self._name}\x1b[0m finished.")

        if iterator_len > 1 or "Tuned" not in self._name:
            logger.info("\x1b[1m{}\x1b[0m fitting and predicting completed".format(self._name))
        # return preds_ds
        return BoostLGBMTransformer(fitted_estimator=self)


class BoostLGBMTransformer(Transformer, MLWritable):

    def __init__(self, fitted_estimator):
        super().__init__()
        self._fitted_estimator = fitted_estimator
        self.models = self._fitted_estimator.models

    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return TmpСommonMLWriter(self.uid)

    def _get_predict_column(self, model: SparkMLModel) -> str:
        # TODO SPARK-LAMA: Rewrite using class recognition.
        try:
            return model.getPredictionCol()
        except AttributeError:
            if isinstance(model, PipelineModel):
                return model.stages[-1].getPredictionCol()

            raise TypeError("Unknown model type! Unable ro retrieve prediction column")

    def predict_single_fold(self,
                            dataset: SparkDataFrame,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        assembler = VectorAssembler(
            inputCols=self._fitted_estimator.getInputCols(),
            outputCol=f"{self._fitted_estimator._name}_vassembler_features",
            handleInvalid="keep"
        )

        temp_sdf = assembler.transform(dataset)

        pred = model.transform(temp_sdf)

        return pred

    def _average_predictions(self, 
                            preds_ds: SparkDataFrame,
                            preds_dfs: List[SparkDataFrame], 
                            pred_col_prefix: str) -> SparkDataFrame:
        # TODO: SPARK-LAMA probably one may write a scala udf function to join multiple arrays/vectors into the one
        # TODO: reg and binary cases probably should be treated without arrays summation
        # we need counter here for EACH row, because for some models there may be no predictions
        # for some rows that means:
        # 1. we need to do left_outer instead of inner join (because the right frame may not contain all rows)
        # 2. we would like to find a mean prediction for each row, but the number of predictiosn may be variable,
        #    that is why we need a counter for each row
        # 3. this counter should depend on if there is None for the right row or not
        # 4. we also need to substitute None's of the right dataframe with empty arrays
        #    to provide uniformity for summing operations
        # 5. we also convert output from vector to an array to combine them
        counter_col_name = "counter"

        if self._fitted_estimator._task_name == "multiclass":
            empty_pred = F.array(*[F.lit(0) for _ in range(self._fitted_estimator.n_classes)])

            def convert_col(prediction_column: str) -> Column:
                return vector_to_array(F.col(prediction_column))

            # full_preds_df
            def sum_predictions_col() -> Column:
                # curr_df[pred_col_prefix]
                return F.transform(
                    F.arrays_zip(pred_col_prefix, f"{pred_col_prefix}_{i}"),
                    lambda x, y: x + y
                ).alias(pred_col_prefix)

            def avg_preds_sum_col() -> Column:
                return array_to_vector(
                    F.transform(pred_col_prefix, lambda x: x / F.col("counter"))
                ).alias(pred_col_prefix)
        else:
            empty_pred = F.lit(0)

            # trivial operator in this case
            def convert_col(prediction_column: str) -> Column:
                return F.col(prediction_column)

            def sum_predictions_col() -> Column:
                # curr_df[pred_col_prefix]
                return (F.col(pred_col_prefix) + F.col(f"{pred_col_prefix}_{i}")).alias(pred_col_prefix)

            def avg_preds_sum_col() -> Column:
                return (F.col(pred_col_prefix) / F.col("counter")).alias(pred_col_prefix)

        full_preds_df = preds_ds.select(
            SparkDataset.ID_COLUMN,
            F.lit(0).alias(counter_col_name),
            empty_pred.alias(pred_col_prefix)
        )
        for i, pred_df in enumerate(preds_dfs):
            pred_col = f"{pred_col_prefix}_{i}"
            full_preds_df = (
                full_preds_df
                .join(pred_df, on=SparkDataset.ID_COLUMN, how="left_outer")
                .select(
                    full_preds_df[SparkDataset.ID_COLUMN],
                    pred_col_prefix,
                    F.when(F.col(pred_col).isNull(), empty_pred)
                        .otherwise(convert_col(pred_col)).alias(pred_col),
                    F.when(F.col(pred_col).isNull(), F.col(counter_col_name))
                        .otherwise(F.col(counter_col_name) + 1).alias(counter_col_name)
                )
                .select(
                    full_preds_df[SparkDataset.ID_COLUMN],
                    counter_col_name,
                    sum_predictions_col()
                )
            )

        full_preds_df = full_preds_df.select(
            SparkDataset.ID_COLUMN,
            avg_preds_sum_col()
        )

        return full_preds_df

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        """Mean prediction for all fitted models.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predicted values.

        """
        assert self.models != [], "Should be fitted first."

        pred_col_prefix = self._fitted_estimator._predict_feature_name()
        preds_dfs = [
            self.predict_single_fold(dataset=dataset, model=model).select(
                SparkDataset.ID_COLUMN,
                F.col(self._get_predict_column(model)).alias(f"{pred_col_prefix}_{i}")
            ) for i, model in enumerate(self.models)
        ]

        predict_sdf = self._average_predictions(dataset, preds_dfs, pred_col_prefix)

        # preds_ds = dataset.empty()
        # preds_ds = self._set_prediction(preds_ds, predict_sdf)

        # return preds_ds
        return predict_sdf
