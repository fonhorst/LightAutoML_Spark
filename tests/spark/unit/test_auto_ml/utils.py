from copy import copy
from typing import List, cast, Optional, Any

from pyspark.ml import Transformer
from pyspark.ml.functions import vector_to_array, array_to_vector

from lightautoml.automl.blend import WeightedBlender
from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.base import UserDefinedRolesDict
from lightautoml.reader.tabular_batch_generator import ReadableToDf
from lightautoml.spark.automl.blend import SparkWeightedBlender
from lightautoml.spark.automl.presets.base import SparkAutoMLPreset
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator

import pyspark.sql.functions as F
import numpy as np


class FakeOpTransformer(Transformer):
    def __init__(self, cols_to_generate: List[str], n_classes: int):
        super().__init__()
        self._cos_to_generate = cols_to_generate
        self._n_classes = n_classes

    def _transform(self, dataset):
        return dataset.select('*', [F.array(*[F.rand() for i in range(self._n_classes)]).alias(f) for f in self._cos_to_generate])


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


class DummySparkMLPipeline(SparkMLPipeline):
    def __init__(
        self,
        cacher_key: str = "",
        name: str = "dummy_pipe"
    ):
        super().__init__(cacher_key, [], force_calc=[True], name=name)

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
                array_to_vector(F.array(*[F.lit(i*10 + j) for j in range(n_classes)])).alias(name)
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
    def __init__(self):
        config_path = '/home/nikolay/wspace/LightAutoML/lightautoml/spark/automl/presets/tabular_config.yml'
        super().__init__(SparkTask("multiclass"), config_path=config_path)

    def create_automl(self, **fit_args):
        # initialize
        reader = DummyReader(self.task)

        cacher_key = "main_cache"
        first_level = [DummySparkMLPipeline(cacher_key, name=f"Lvl_0_Pipe_{i}") for i in range(3)]
        second_level = [DummySparkMLPipeline(cacher_key, name=f"Lvl_1_Pipe_{i}") for i in range(2)]
        levels = [first_level, second_level]

        blender = SparkWeightedBlender()

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
