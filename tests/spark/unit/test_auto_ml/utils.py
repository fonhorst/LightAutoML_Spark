from typing import List, cast, Optional, Any

from pyspark.ml import Transformer

from lightautoml.automl.blend import WeightedBlender
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
        return super().fit_read(train_data, features_names, roles, **kwargs)

    def read(self, data: SparkDataFrame, features_names: Any = None, add_array_attrs: bool = False) -> SparkDataset:
        return super().read(data, features_names, add_array_attrs)


class DummySparkMLPipeline(SparkMLPipeline):
    def __init__(
        self,
        cacher_key: str = "",
        name: str = "dummy_pipe"
    ):
        super().__init__(cacher_key, [], name=name)

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
        sdf = sdf.select('*', *self._output_roles.keys())

        out_val_ds = cast(SparkDataset, val_ds.empty())
        out_val_ds.set_data(sdf, list(self._output_roles.keys()), self._output_roles)

        return out_val_ds


class DummyTabularAutoML(SparkAutoMLPreset):
    def __init__(self):
        super().__init__(SparkTask("multiclass"))

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
