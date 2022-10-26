from abc import ABC, abstractmethod
from typing import Optional

from lightautoml.dataset.base import RolesDict
from pyspark.ml import Transformer

from sparklightautoml.dataset.base import SparkDataset


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
        sdf = self.transformer(*args, **kwargs).transform(dataset.data)

        roles = {**dataset.roles, **self.output_roles}

        out_ds = dataset.empty()
        out_ds.set_data(sdf, list(roles.keys()), roles)

        return out_ds
