from abc import ABC, abstractmethod
from typing import Optional

from lightautoml.dataset.base import RolesDict
from pyspark.ml import Transformer
from pyspark.ml.pipeline import PipelineModel

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.utils import ColumnsSelectorTransformer


class TransformerInputOutputRoles(ABC):
    """
    Class that represents input features and input roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def input_roles(self) -> Optional[RolesDict]:
        """Returns dict of input roles"""
        ...

    @property
    @abstractmethod
    def output_roles(self) -> Optional[RolesDict]:
        """Returns dict of output roles"""
        ...

    @abstractmethod
    def transformer(self, *args, **kwargs) -> Optional[Transformer]:
        ...

    def _make_transformed_dataset(self, dataset: SparkDataset, *args, **kwargs) -> SparkDataset:
        sdf = PipelineModel(stages=[
            self.transformer(*args, **kwargs),
            ColumnsSelectorTransformer(input_cols=list(self.output_roles.keys()), optional_cols=dataset.service_columns)
        ]).transform(dataset.data)

        roles = {**self.output_roles}

        out_ds = dataset.empty()
        out_ds.set_data(sdf, list(roles.keys()), roles)

        return out_ds
