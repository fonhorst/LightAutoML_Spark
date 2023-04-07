import logging
import math
import multiprocessing
import warnings
from copy import copy
from typing import Dict, Optional, Tuple, Union, cast, List

import lightgbm as lgb
import pandas as pd
import pyspark.sql.functions as sf
from lightautoml.dataset.roles import ColumnRole
from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.utils.timer import TaskTimer
from lightautoml.validation.base import TrainValidIterator
from lightgbm import Booster
from pandas import Series
from pyspark.ml import Transformer, PipelineModel, Model
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.util import MLWritable, MLReadable, MLWriter
from synapse.ml.lightgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRegressionModel,
    LightGBMClassificationModel,
)
from synapse.ml.onnx import ONNXModel

from sparklightautoml.computations.manager import computations_manager, PoolType, LGBMDatasetSlot
from sparklightautoml.dataset.base import SparkDataset, PersistenceManager
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from sparklightautoml.mlwriters import (
    LightGBMModelWrapperMLReader,
    LightGBMModelWrapperMLWriter,
    ONNXModelWrapperMLReader,
    ONNXModelWrapperMLWriter,
)
from sparklightautoml.transformers.base import (
    DropColumnsTransformer,
    PredictionColsTransformer,
    ProbabilityColsTransformer,
)
from sparklightautoml.transformers.scala_wrappers.balanced_union_partitions_coalescer import \
    BalancedUnionPartitionsCoalescerTransformer
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)


class LightGBMModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `synapse.ml.lightgbm.[LightGBMRegressionModel|LightGBMClassificationModel]`
    to fix issue with loading model from saved composite pipeline.

    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: Union[LightGBMRegressionModel, LightGBMClassificationModel] = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return LightGBMModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return LightGBMModelWrapperMLReader()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return self.model.transform(dataset)


class ONNXModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `ONNXModel` to fix issue with loading model from saved composite pipeline.

    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: ONNXModel = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return ONNXModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return ONNXModelWrapperMLReader()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return self.model.transform(dataset)


class SparkBoostLGBM(SparkTabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in synapse.ml documentation:

    - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMClassifier
    - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMRegressor

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

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
        "minDataInLeaf": 5,
        # e.g. num trees
        "numIterations": 3000,
        "earlyStoppingRound": 50,
        # for regression
        "alpha": 1.0,
        "lambdaL1": 0.0,
        "lambdaL2": 0.0,
    }

    # mapping between metric name defined via SparkTask
    # and metric names supported by LightGBM
    _metric2lgbm = {
        "binary": {"auc": "auc", "aupr": "areaUnderPR"},
        "reg": {
            "r2": "rmse",
            "mse": "mse",
            "mae": "mae",
        },
        "multiclass": {"crossentropy": "cross_entropy"},
    }

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = None,
        use_single_dataset_mode: bool = True,
        max_validation_size: int = 10_000,
        chunk_size: int = 4_000_000,
        convert_to_onnx: bool = False,
        mini_batch_size: int = 5000,
        seed: int = 42,
        parallelism: int = 1,
        experimental_parallel_mode: bool = False
    ):
        optimization_search_space = optimization_search_space if optimization_search_space else dict()
        SparkTabularMLAlgo.__init__(self, default_params, freeze_defaults,
                                    timer, optimization_search_space, parallelism)
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"
        self._raw_prediction_col_name = "raw_prediction"
        self._assembler = None
        self._drop_cols_transformer = None
        self._use_single_dataset_mode = use_single_dataset_mode
        self._max_validation_size = max_validation_size
        self._seed = seed
        self._models_feature_importances = []
        self._chunk_size = chunk_size
        self._convert_to_onnx = convert_to_onnx
        self._mini_batch_size = mini_batch_size
        self._experimental_parallel_mode = experimental_parallel_mode

    def _infer_params(self) -> Tuple[dict, int]:
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

        if task == "reg":
            params["objective"] = "regression"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "binary":
            params["objective"] = "binary"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task}")

        if task != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        params = {**params}

        return params, verbose_eval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        self.task = train_valid_iterator.train.task

        sds = cast(SparkDataset, train_valid_iterator.train)
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
            ntrees = 2000
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
        """Train on train dataset and predict on holdout dataset.

        Args:.
            suggested_params: suggested params
            estimated_n_trials: Number of trials.

        Returns:
            Target predictions for valid dataset.

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
            low=4,
            high=255,
        )

        if self.task.name == "binary" or self.task.name == "multiclass":
            optimization_search_space["isUnbalance"] = SearchSpace(Distribution.DISCRETEUNIFORM, low=0, high=1, q=1)

        if estimated_n_trials > 30:
            optimization_search_space["baggingFraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            optimization_search_space["minSumHessianInLeaf"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-3,
                high=10.0,
            )

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

    def predict_single_fold(
        self, dataset: SparkDataset, model: Union[LightGBMRegressor, LightGBMClassifier]
    ) -> SparkDataFrame:

        temp_sdf = self._assembler.transform(dataset.data)

        pred = model.transform(temp_sdf)

        return pred

    def _get_num_threads(self, train: SparkDataFrame):
        master_addr = train.spark_session.conf.get("spark.master")
        if master_addr.startswith("local-cluster"):
            # exec_str, cores_str, mem_mb_str
            _, cores_str, _ = master_addr[len("local-cluster["): -1].split(",")
            cores = int(cores_str)
            num_threads = max(cores - 1, 1)
        elif master_addr.startswith("local"):
            cores_str = master_addr[len("local["): -1]
            cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
            num_threads = max(cores - 1, 1)
        else:
            num_threads = max(int(train.spark_session.conf.get("spark.executor.cores", "1")) - 1, 1)

        return num_threads

    def _do_convert_to_onnx(self, train: SparkDataset, ml_model):
        logger.info("Model convert is started")
        booster_model_str = ml_model.getLightGBMBooster().modelStr().get()
        booster = lgb.Booster(model_str=booster_model_str)
        model_payload_ml = self._convert_model(booster, len(train.features))

        onnx_ml = ONNXModel().setModelPayload(model_payload_ml)

        if train.task.name == "reg":
            onnx_ml = (
                onnx_ml.setDeviceType("CPU")
                    .setFeedDict({"input": f"{self._name}_vassembler_features"})
                    .setFetchDict({ml_model.getPredictionCol(): "variable"})
                    .setMiniBatchSize(self._mini_batch_size)
            )
        else:
            onnx_ml = (
                onnx_ml.setDeviceType("CPU")
                    .setFeedDict({"input": f"{self._name}_vassembler_features"})
                    .setFetchDict({ml_model.getProbabilityCol(): "probabilities", ml_model.getPredictionCol(): "label"})
                    .setMiniBatchSize(self._mini_batch_size)
            )

        logger.info("Model convert is ended")

        return onnx_ml

    def _merge_train_val(self, train: SparkDataset, valid: SparkDataset) -> SparkDataFrame:
        train_data = train.data
        valid_data = valid.data
        valid_size = valid.data.count()
        max_val_size = self._max_validation_size
        if valid_size > max_val_size:
            warnings.warn(
                f"Maximum validation size for SparkBoostLGBM is exceeded: {valid_size} > {max_val_size}. "
                f"Reducing validation size down to maximum.",
                category=RuntimeWarning,
            )

            valid_data = valid_data.sample(fraction=max_val_size / valid_size, seed=self._seed)

        td = train_data.select('*', sf.lit(False).alias(self.validation_column))
        vd = valid_data.select('*', sf.lit(True).alias(self.validation_column))
        full_data = td.unionByName(vd)

        if train_data.rdd.getNumPartitions() == valid_data.rdd.getNumPartitions():
            full_data = BalancedUnionPartitionsCoalescerTransformer().transform(full_data)
        else:
            message = f"Cannot apply BalancedUnionPartitionsCoalescer " \
                      f"due to train and val datasets doesn't have the same number of partitions. " \
                      f"The train dataset has {train_data.rdd.getNumPartitions()} partitions " \
                      f"while the val dataset has {valid_data.rdd.getNumPartitions()} partitions." \
                      f"Continue with plain union." \
                      f"In some situations it may negatively affect behavior of SynapseML LightGBM " \
                      f"due to empty partitions"
            warnings.warn(message, RuntimeWarning)

        return full_data

    def fit_predict_single_fold(
        self,
            fold_prediction_column: str,
            train: SparkDataset,
            valid: Optional[SparkDataset] = None,
            slot: Optional[LGBMDatasetSlot] = None
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        if self.task is None:
            self.task = train.task

        (params, verbose_eval) = self._infer_params()

        logger.info(f"Input cols for the vector assembler: {train.features}")
        logger.info(f"Running lgb with the following params: {params}")

        if train.task.name in ["binary", "multiclass"]:
            params["rawPredictionCol"] = self._raw_prediction_col_name
            params["probabilityCol"] = fold_prediction_column
            params["predictionCol"] = self._prediction_col_name
            params["isUnbalance"] = True
        else:
            params["predictionCol"] = fold_prediction_column

        if valid is not None:
            full_data = self._merge_train_val(train, valid)
        else:
            train_data = train.data
            assert self.validation_column in train_data.columns
            # TODO: make filtering of excessive valid dataset
            full_data = train_data

        if slot is not None:
            params["numTasks"] = slot.num_tasks
            params["numThreads"] = slot.num_threads
            params["useBarrierExecutionMode"] = slot.use_barrier_execution_mode
        else:
            params["numThreads"] = self._get_num_threads(train.data)

        # prepare assembler
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=train.features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        # build the booster
        lgbm_booster = LightGBMRegressor if train.task.name == "reg" else LightGBMClassifier

        lgbm = lgbm_booster(
            **params,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=train.target_column,
            validationIndicatorCol=self.validation_column,
            verbosity=verbose_eval,
            useSingleDatasetMode=self._use_single_dataset_mode,
            isProvideTrainingMetric=True,
            chunkSize=self._chunk_size,
        )

        if train.task.name == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        logger.info(f"Use single dataset mode: {lgbm.getUseSingleDatasetMode()}. NumThreads: {lgbm.getNumThreads()}")

        # fitting the model
        ml_model = lgbm.fit(self._assembler.transform(full_data))

        # handle the model
        ml_model = self._do_convert_to_onnx(train, ml_model) if self._convert_to_onnx else ml_model
        self._models_feature_importances.append(ml_model.getFeatureImportances(importance_type="gain"))

        # predict validation
        val_pred = ml_model.transform(self._assembler.transform(valid.data))
        val_pred = DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name, self._raw_prediction_col_name],
        ).transform(val_pred)

        return ml_model, val_pred, fold_prediction_column

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        logger.info("Starting LGBM fit")
        self.fit_predict(train_valid)
        logger.info("Finished LGBM fit")

    def get_features_score(self) -> Series:
        imp = 0
        for model_feature_impotances in self._models_feature_importances:
            imp = imp + pd.Series(model_feature_impotances)

        imp = imp / len(self._models_feature_importances)

        def flatten_features(feat: str):
            role = self.input_roles[feat]
            if isinstance(role, NumericVectorOrArrayRole):
                return [f"{feat}_pos_{i}" for i in range(role.size)]
            return [feat]

        index = [
            ff
            for feat in self._assembler.getInputCols()
            for ff in flatten_features(feat)
        ]

        result = Series(list(imp), index=index).sort_values(ascending=False)
        return result

    @staticmethod
    def _convert_model(lgbm_model: Booster, input_size: int) -> bytes:
        from onnxmltools.convert import convert_lightgbm
        from onnxconverter_common.data_types import FloatTensorType

        initial_types = [("input", FloatTensorType([-1, input_size]))]
        onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset=9)
        return onnx_model.SerializeToString()

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        if self._convert_to_onnx:
            wrapped_models = [ONNXModelWrapper(m) for m in self.models]
        else:
            wrapped_models = [LightGBMModelWrapper(m) for m in self.models]
        models: List[Transformer] = [
            el
            for m in wrapped_models
            for el in [
                m,
                DropColumnsTransformer(
                    remove_cols=[],
                    optional_remove_cols=[
                        self._prediction_col_name,
                        self._probability_col_name,
                        self._raw_prediction_col_name,
                    ],
                ),
            ]
        ]
        if self._convert_to_onnx:
            if self.task.name in ["binary", "multiclass"]:
                models.append(
                    ProbabilityColsTransformer(
                        probability_cols=self._models_prediction_columns, num_classes=self.n_classes
                    )
                )
            else:
                models.append(PredictionColsTransformer(prediction_cols=self._models_prediction_columns))
        averaging_model = PipelineModel(stages=[
            self._assembler,
            *models,
            avr,
            self._build_vector_size_hint(self.prediction_feature, self.prediction_role)
        ])
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=[self._assembler.getOutputCol()] + self._models_prediction_columns,
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes,
        )
        return avr

    def fit_predict(self, train_valid_iterator: SparkBaseTrainValidIterator) -> SparkDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``numpy.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        logger.info("Starting LGBM fit")
        self.timer.start()

        res = super().fit_predict(train_valid_iterator)

        logger.info("Finished LGBM fit")
        return res

    def _prepare_trains(self, paralellism_mode, train_df, max_job_parallelism: int) -> List[DatasetSlot]:
        if self.parallelism_mode == ParallelismMode.pref_locs:
            execs_per_job = max(1, math.floor(len(self._executors) / max_job_parallelism))

            if len(self._executors) % max_job_parallelism != 0:
                warnings.warn(f"Uneven number of executors per job. Setting execs per job: {execs_per_job}.")

            slots_num = int(len(self._executors) / execs_per_job)

            _train_slots = []

            def _coalesce_df_to_locs(df: SparkDataFrame, pref_locs: List[str]):
                df = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs).transform(df)
                df = df.cache()
                df.write.mode('overwrite').format('noop').save()
                return df

            for i in range(slots_num):
                pref_locs = self._executors[i * execs_per_job: (i + 1) * execs_per_job]

                # prepare train
                train_df = _coalesce_df_to_locs(train_df, pref_locs)

                # prepare test
                test_df = _coalesce_df_to_locs(test_df, pref_locs)

                _train_slots.append(DatasetSlot(
                    train_df=train_df,
                    test_df=test_df,
                    pref_locs=pref_locs,
                    num_tasks=len(pref_locs) * self._cores_per_exec,
                    num_threads=-1,
                    use_single_dataset_mode=True,
                    free=True
                ))

                print(f"Pref lcos for slot #{i}: {pref_locs}")
        elif self.parallelism_mode == ParallelismMode.no_single_dataset_mode :
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=-1,
                use_single_dataset_mode=False,
                free=False
            )]
        elif self.parallelism_mode == ParallelismMode.single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            num_threads_per_exec = max(1, math.floor(num_tasks_per_job / len(self._executors)))

            if num_threads_per_exec != 1:
                warnings.warn(f"Num threads per exec {num_threads_per_exec} != 1. "
                              f"Overcommitting or undercommiting may happen due to "
                              f"uneven allocations of cores between executors for a job")

            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=num_threads_per_exec,
                use_single_dataset_mode=True,
                free=False
            )]
        else:
            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=self.train_dataset.rdd.getNumPartitions(),
                num_threads=-1,
                use_single_dataset_mode=True,
                free=False
            )]

        return _train_slots

    def _parallel_fit(self, parallelism: int, train_valid_iterator: SparkBaseTrainValidIterator) -> Tuple[
        List[Model], List[SparkDataFrame], List[str]]:
        # TODO: prepare train_valid_iterator
        # 1. check if we need to run in experimental parallel mode
        # 2. lock further running and prepare the train valid iteratorS
        if self._experimental_parallel_mode:
            # TODO: 1. locking by the same lock
            # TODO: is it possible to run exclusively or not?
            # TODO: 2. create slot-based train_val_iterator
            # TODO: 3. Redefine params through setInternalParallelismParams(...) which is added with redefined infer_params(...)



            dataset = train_valid_iterator.train_val_single_dataset

            slots: List[LGBMDatasetSlot] = self._prepare_trains(paralellism_mode=, train_df=dataset.data, max_job_parallelism=)

            # 1. take dataset from train_valid_iterator
            # 2. squash it for train_valid
            # 3. convert it to a new train_valid iterator

            num_folds = len(train_valid_iterator)

            def build_fit_func(i: int, timer: TaskTimer, mdl_pred_col: str):
                def func(slot: LGBMDatasetSlot) -> Optional[Tuple[int, Model, SparkDataFrame, str]]:
                    if num_folds > 1:
                        logger.info2(
                            "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m "
                            "=====".format(i, self._name)
                        )

                    # TODO: need to filter it according to fold or is_val sign
                    ds = slot.dataset

                    mdl, vpred, _ = self.fit_predict_single_fold(mdl_pred_col, ds, slot=slot)
                    vpred = vpred.select(SparkDataset.ID_COLUMN, ds.target_column, mdl_pred_col)

                    timer.write_run_info()

                    return i, mdl, vpred, mdl_pred_col

                return func

            fit_tasks = [
                build_fit_func(i, copy(self.timer), f"{self.prediction_feature}_{i}")
                for i, _ in enumerate(train_valid_iterator)
            ]

            manager = computations_manager()
            results = manager.compute_with_slots(
                name=self.name,
                slots=slots,
                tasks=fit_tasks,
                pool_type=PoolType.DEFAULT
            )

            models = [model for _, model, _, _ in results]
            val_preds = [val_pred for _, _, val_pred, _ in results]
            model_prediction_cols = [model_prediction_col for _, _, _, model_prediction_col in results]

            # unprepare dataset

            return models, val_preds, model_prediction_cols

        return super()._parallel_fit(parallelism, train_valid_iterator)
