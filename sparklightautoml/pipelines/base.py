from abc import ABC, abstractmethod
from typing import List

from lightautoml.dataset.base import RolesDict
from pyspark.ml import Transformer


class TransformerInputOutputRoles(ABC):
    """
    Class that represents input features and input roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def input_roles(self) -> RolesDict:
        """Returns dict of input roles"""
        ...

    @property
    @abstractmethod
    def output_roles(self) -> RolesDict:
        """Returns dict of output roles"""
        ...

    @abstractmethod
    def transformer(self, *args, **kwargs) -> Transformer:
        ...
