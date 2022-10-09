from copy import copy
from typing import List, cast, Optional, Any, Tuple, Callable

import numpy as np
import pyspark.sql.functions as sf
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.functions import array_to_vector

from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.base import UserDefinedRolesDict
from lightautoml.reader.tabular_batch_generator import ReadableToDf
from sparklightautoml.automl.blend import SparkWeightedBlender
from sparklightautoml.automl.presets.base import SparkAutoMLPreset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.caching import CacheManager
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.iterators import SparkFoldsIterator


class FakeOpTransformer(Transformer):
    def __init__(self, cols_to_generate: List[str], n_classes: int):
        super().__init__()
        self._cos_to_generate = cols_to_generate
        self._n_classes = n_classes

    def _transform(self, dataset):
        return dataset.select(
            '*',
            *[
                array_to_vector(sf.array(*[sf.rand() for _ in range(self._n_classes)])).alias(f)
                for f in self._cos_to_generate
            ]
        )


class DummyReader(SparkToSparkReader):
    def __init__(self, task: SparkTask):
        super().__init__(task)

    def fit_read(self, train_data: SparkDataFrame, features_names: Any = None, roles: UserDefinedRolesDict = None,
                 **kwargs: Any) -> SparkDataset:

        self.target_col = roles["target"]
        self._roles = {c: NumericRole() for c in train_data.columns if c != self.target_col}

        train_data = self._create_unique_ids(train_data, cacher_key='main_cache')
        train_data, folds_col = self._create_folds(train_data, kwargs={})

        sds = SparkDataset(train_data, self._roles, task=self.task, target=self.target_col, folds=folds_col)
        return sds

    def read(self, data: SparkDataFrame, features_names: Any = None, add_array_attrs: bool = False) -> SparkDataset:
        data = self._create_unique_ids(data, cacher_key='main_cache')
        sds = SparkDataset(data, self._roles, task=self.task, target=self.target_col)
        return sds


class DummyMLAlgo(SparkTabularMLAlgo):
    _name = "dummy"

    def __init__(self, n_classes: int, name: str):
        self._name = name
        super().__init__(cacher_key='dummy_ml_algo')
        self.n_classes = n_classes

    def fit_predict_single_fold(self, fold_prediction_column: str, full: SparkDataset, train: SparkDataset,
                                valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        prediction = array_to_vector(sf.array(*[sf.lit(10 * 10 + j) for j in range(self.n_classes)]))\
            .alias(fold_prediction_column)
        pred_df = valid.data.select('*', prediction)

        fake_op = FakeOpTransformer(cols_to_generate=[fold_prediction_column], n_classes=self.n_classes)
        ml_model = PipelineModel(stages=[fake_op])

        return ml_model, pred_df, fold_prediction_column

    def predict_single_fold(self, model: SparkMLModel, dataset: SparkDataset) -> SparkDataFrame:
        raise NotImplementedError("")

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        fake_ops: List[Transformer] = [
            FakeOpTransformer(cols_to_generate=[mcol], n_classes=self.n_classes)
            for mcol in self._models_prediction_columns
        ]
        averaging_model = PipelineModel(stages=fake_ops + [avr])
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=self._models_prediction_columns,
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes
        )
        return avr


class DummySparkMLPipeline(SparkMLPipeline):
    def __init__(
        self,
        cache_manager: CacheManager,
        name: str = "dummy_pipe"
    ):
        super().__init__(cache_manager, [], force_calc=[True], name=name)

    def fit_predict(self, train_valid: SparkBaseTrainValidIterator) -> SparkDataset:
        val_ds = train_valid.get_validation_data()

        n_classes = 10

        self._output_roles = {
            f"predictions_{self.name}_alg_{i}":
                NumericVectorOrArrayRole(size=n_classes,
                                         element_col_name_template=f"{self.name}_alg_{i}" + "_{}",
                                         dtype=np.float32,
                                         force_input=True,
                                         prob=False)
            for i in range(3)
        }

        self._transformer = FakeOpTransformer(cols_to_generate=self.output_features, n_classes=n_classes)

        sdf = cast(SparkDataFrame, val_ds.data)
        sdf = sdf.select(
            '*',
            *[
                array_to_vector(sf.array(*[sf.lit(i * 10 + j) for j in range(n_classes)])).alias(name)
                for i, name in enumerate(self._output_roles.keys())
            ]
        )

        out_roles = copy(self._output_roles)
        out_roles.update(train_valid.train.roles)
        out_roles.update(train_valid.input_roles)

        out_val_ds = cast(SparkDataset, val_ds.empty())
        out_val_ds.set_data(sdf, list(out_roles.keys()), out_roles)

        return out_val_ds


class DummyTabularAutoML(SparkAutoMLPreset):
    def __init__(self, n_classes: int):
        config_path = '/home/nikolay/wspace/LightAutoML/lightautoml/spark/automl/presets/tabular_config.yml'
        super().__init__(SparkTask("multiclass"), config_path=config_path)
        self._n_classes = n_classes

    def _create_validation_iterator(self, train: SparkDataset, valid: Optional[SparkDataset], n_folds: Optional[int],
                                    cv_iter: Optional[Callable]) -> SparkBaseTrainValidIterator:
        return SparkFoldsIterator(train, n_folds)

    def create_automl(self, **fit_args):
        # initialize
        reader = DummyReader(self.task)

        cache_manager = CacheManager()

        # first_level = [DummySparkMLPipeline(cacher_key, name=f"Lvl_0_Pipe_{i}") for i in range(3)]
        # second_level = [DummySparkMLPipeline(cacher_key, name=f"Lvl_1_Pipe_{i}") for i in range(2)]

        first_level = [
            SparkMLPipeline(cache_manager, ml_algos=[DummyMLAlgo(self._n_classes, name=f"dummy_0_{i}")])
            for i in range(3)
        ]
        second_level = [
            SparkMLPipeline(cache_manager, ml_algos=[DummyMLAlgo(self._n_classes, name=f"dummy_1_{i}")])
            for i in range(1)
        ]

        levels = [first_level, second_level]

        blender = SparkWeightedBlender(max_iters=0, max_inner_iters=1)

        self._initialize(
            reader,
            levels,
            skip_conn=True,
            blender=blender,
            return_all_predictions=False,
            timer=self.timer,
        )

    def get_individual_pdp(self, test_data: ReadableToDf, feature_name: str, n_bins: Optional[int] = 30,
                           top_n_categories: Optional[int] = 10, datetime_level: Optional[str] = "year"):
        raise ValueError("Not supported")

    def plot_pdp(self, test_data: ReadableToDf, feature_name: str, individual: Optional[bool] = False,
                 n_bins: Optional[int] = 30, top_n_categories: Optional[int] = 10, top_n_classes: Optional[int] = 10,
                 datetime_level: Optional[str] = "year"):
        raise ValueError("Not supported")
