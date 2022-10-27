from abc import ABC, abstractmethod
from typing import Optional, List

from lightautoml.dataset.base import RolesDict
from pyspark.ml import Transformer
from pyspark.ml.pipeline import PipelineModel

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.utils import ColumnsSelectorTransformer, WrappingSelectingPipelineModel


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

    def transformer(self, *args, **kwargs) -> Optional[Transformer]:
        transformer = self._build_transformer(*args, **kwargs)
        return self._clean_transformer_columns(transformer, self._get_service_columns()) if transformer else None

    @abstractmethod
    def _get_service_columns(self) -> List[str]:
        ...

    @abstractmethod
    def _build_transformer(self, *args, **kwargs) -> Optional[Transformer]:
        ...

    def _clean_transformer_columns(self, transformer: Transformer, service_columns: Optional[List[str]] = None):
        # we don't service_columns cause they should be available in the input by default
        return WrappingSelectingPipelineModel(
            stages=[transformer],
            input_columns=list(self.output_roles.keys())
        )

    def _make_transformed_dataset(self, dataset: SparkDataset, *args, **kwargs) -> SparkDataset:
        roles = {**self.output_roles}

        sdf = PipelineModel(stages=[
            self.transformer(*args, **kwargs),
            ColumnsSelectorTransformer(input_cols=list(roles.keys()), optional_cols=dataset.service_columns)
        ]).transform(dataset.data)

        out_ds = dataset.empty()
        out_ds.set_data(sdf, list(roles.keys()), roles)

        return out_ds
