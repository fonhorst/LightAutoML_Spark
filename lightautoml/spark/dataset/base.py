from contextlib import contextmanager
from copy import copy
from typing import Sequence, Any, Tuple, Union, Optional, NewType, List, cast, Dict, Set

import pandas as pd
import numpy as np
import pyspark
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, Column
from pyspark.sql.session import SparkSession

from lightautoml.dataset.base import LAMLDataset, IntIdx, RowSlice, ColSlice, LAMLColumn, RolesDict, \
    valid_array_attributes, array_attr_roles
from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset, NpRoles
from lightautoml.dataset.roles import ColumnRole, NumericRole, DropRole
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.tasks import Task

# SparkDataFrame = NewType('SparkDataFrame', pyspark.sql.DataFrame)
SparkDataFrame = pyspark.sql.DataFrame


class SparkDataset(LAMLDataset):
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "SparkDataset"

    ID_COLUMN = "_id"

    def empty(self) -> "SparkDataset":

        dataset = cast(SparkDataset, super().empty())
        dataset._dependencies = []

        return dataset

    @staticmethod
    def uncache_dataframes():
        spark = SparkSession.getActiveSession()
        tables = spark.catalog.listTables()
        for table in tables:
            spark.catalog.uncacheTable(table.name)

    @classmethod
    def concatenate(cls, datasets: Sequence["SparkDataset"]) -> "SparkDataset":
        """
        Concat multiple datasets by joining their internal pyspark.sql.DataFrame
        using inner join on special hidden '_id' column
        Args:
            datasets: spark datasets to be joined

        Returns:
            a joined dataset, containing features (and columns too) from all datasets
            except containing only one _id column
        """
        assert len(datasets) > 0, "Cannot join an empty list of datasets"

        # requires presence of hidden "_id" column in each dataset
        # that should be saved across all transformations
        features = [feat for ds in datasets for feat in ds.features]
        roles = {col: role for ds in datasets for col, role in ds.roles.items()}
        curr_sdf = datasets[0].data

        for ds in datasets[1:]:
            curr_sdf = curr_sdf.join(ds.data, cls.ID_COLUMN)

        # TODO: SPARK-LAMA can we do it without cast?
        curr_sdf = cast(SparkDataFrame, curr_sdf.select(datasets[0].data[cls.ID_COLUMN], *features))

        output = datasets[0].empty()
        output.set_data(curr_sdf, features, roles, dependencies=[d for ds in datasets for d in ds.dependencies])

        return output

    def __init__(self,
                 data: SparkDataFrame,
                 roles: Optional[RolesDict],
                 task: Optional[Task] = None,
                 dependencies: Optional[List['SparkDataset']] = None,
                 **kwargs: Any):

        if "target" in kwargs:
            # assert "target" in kwargs, "Arguments should contain 'target'"
            assert isinstance(kwargs["target"], pyspark.sql.DataFrame), "Target should be a spark dataframe"

            target_sdf = cast(pyspark.sql.DataFrame, kwargs["target"])

            assert len(target_sdf.columns) == 2, "Only 2 columns should be in the target spark dataframe"
            assert SparkDataset.ID_COLUMN in target_sdf.columns, \
                f"id column {SparkDataset.ID_COLUMN} should be presented in the target spark dataframe"

            self._target_column: str = next(c for c in target_sdf.columns if c != SparkDataset.ID_COLUMN)

        self._folds_column = None
        if "folds" in kwargs:
            # folds = kwargs["folds"]
            # assert isinstance(folds, list)
            # correct_folds = (
            #     (self.ID_COLUMN in train.columns) and (self.ID_COLUMN in val.columns)
            #     for train, val in folds
            # )
            # assert all(correct_folds), "ID_COLUMN should be presented everywhere"

            assert isinstance(kwargs["folds"], pyspark.sql.DataFrame), "Folds should be a spark dataframe"

            folds_sdf = cast(SparkDataFrame, kwargs["folds"])

            assert len(folds_sdf.columns) == 2, "Only 2 columns should be in the folds spark dataframe"
            assert SparkDataset.ID_COLUMN in folds_sdf.columns, \
                f"id column {SparkDataset.ID_COLUMN} should be presented in the folds spark dataframe"
            self._folds_column: str = next(c for c in folds_sdf.columns if c != SparkDataset.ID_COLUMN)

        # TODO: SPARK-LAMA there is a clear problem with this target
        #       we either need to bring this column through all datasets(e.g. duplication)
        #       or really save it as a separate dataframe

        self._validate_dataframe(data)

        self._data = None
        self._is_frozen_in_cache: bool = False
        self._dependencies = [] if dependencies is None else dependencies
        self._service_columns: Set[str] = {self.ID_COLUMN}

        roles = roles if roles else dict()

        # currently only target is supported
        # adapted from PandasDataset
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    roles[f] = DropRole()

        super().__init__(data, None, roles, task, **kwargs)

    @property
    def spark_session(self):
        return SparkSession.getActiveSession()

    @property
    def data(self) -> SparkDataFrame:
        return self._data

    @data.setter
    def data(self, val: SparkDataFrame) -> None:
        self._data = val

    @property
    def features(self) -> List[str]:
        """Get list of features.

        Returns:
            list of features.

        """
        return [c for c in self.data.columns if c not in self._service_columns] \
            if self.data else []

    @features.setter
    def features(self, val: None):
        """Ignore setting features.

        Args:
            val: ignored.

        """
        pass
        # raise NotImplementedError("The operation is not supported")

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """Define how to set roles.

        Args:
            val: Roles.

        Note:
            There is different behavior for different type of val parameter:

                - `List` - should be same len as ``data.shape[1]``.
                - `None` - automatic set ``NumericRole(np.float32)``.
                - ``ColumnRole`` - single role for all.
                - ``dict``.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
        elif val:
            role = cast(ColumnRole, val)
            self._roles = dict(((x, role) for x in self.features))
        else:
            raise ValueError()

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        return -1, len(self.features)
        # TODO: we may call count here but it may lead to a huge calculation
        # assert self.data.is_cached, "Shape should not be calculated if data is not cached"
        # return self.data.count(), len(self.data.columns)

    @property
    def target_column(self) -> str:
        return self._target_column

    @property
    def folds_column(self) -> str:
        # raise NotImplementedError("It is unsupported now")
        return self._folds_column

    @property
    def dependencies(self) -> Optional[List['SparkDataset']]:
        return self._dependencies

    @dependencies.setter
    def dependencies(self, val: List['SparkDataset']) -> None:
        assert not self._dependencies
        self._dependencies = val

    @property
    def is_frozen_in_cache(self) -> bool:
        return self._is_frozen_in_cache

    @is_frozen_in_cache.setter
    def is_frozen_in_cache(self, val: bool) -> None:
        self._is_frozen_in_cache = val

    @property
    def service_columns(self) -> List[str]:
        return list(self._service_columns)

    def cache_and_materialize(self) -> None:
        """
        This method caches Spark DataFrame and calls count() to materialize the data in Spark cache
        Be aware it is a blocking operation
        """
        if not self.data.is_cached:
            self.data = self.data.cache()
        self.data.count()

    def cache(self) -> None:
        if not self.data.is_cached:
            self.data = self.data.cache()

    def uncache(self) -> None:
        if not self.is_frozen_in_cache:
            self.data.unpersist()

    def unwind_dependencies(self) -> None:
        """
            Uncache all dependencies, e.g. the parent (or parents) of this spark dataframe.
            Use this method when it is assured that the current dataset has been already materialized.
            (for instance, after a fit method is applied).
        """
        for sds in self._dependencies:
            sds.uncache()

    @contextmanager
    def applying_temporary_caching(self):
        """Performs cache and uncache before and after a code block"""
        if not self.data.is_cached:
            self.data = self.data.cache()

        is_already_frozen = self.is_frozen_in_cache

        if not is_already_frozen:
            self.is_frozen_in_cache = True

        yield self

        if not is_already_frozen:
            self.is_frozen_in_cache = False
            self.data = self.data.unpersist()

    def __repr__(self):
        return f"SparkDataset ({self.data})"

    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        rslice, clice = k

        # TODO: make handling of rslice

        if isinstance(clice, str):
            clice = [clice]

        assert all(c in self.features for c in clice), \
            f"Not all columns presented in the dataset.\n" \
            f"Presented: {self.features}\n" \
            f"Asked for: {clice}"

        sdf = cast(SparkDataFrame, self.data.select(self.ID_COLUMN, *clice))
        roles = {c: self.roles[c] for c in clice}

        output = self.empty()
        output.set_data(sdf, clice, roles)

        return output
        # raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __setitem__(self, k: str, val: Any):
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def _validate_dataframe(self, sdf: SparkDataFrame) -> None:
        assert self.ID_COLUMN in sdf.columns, \
            f"No special unique row id column (the column name: {self.ID_COLUMN}) in the spark dataframe"
        # assert kwargs["target"] in data.columns, \
        #     f"No target column (the column name: {kwargs['target']}) in the spark dataframe"

    def _materialize_to_pandas(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, ColumnRole]]:
        sdf = self.data

        def expand_if_vec_or_arr(col, role) -> Tuple[List[Column], ColumnRole]:
            if not isinstance(role, NumericVectorOrArrayRole):
                return [col], role
            vrole = cast(NumericVectorOrArrayRole, role)

            def to_array(column):
                if vrole.is_vector:
                    return vector_to_array(column)
                return column

            arr = [
                to_array(F.col(col))[i].alias(vrole.feature_name_at(i))
                for i in range(vrole.size)
            ]

            return arr, NumericRole(dtype=vrole.dtype)

        arr_cols = (expand_if_vec_or_arr(c, self.roles[c]) for c in self.features)
        all_cols_and_roles = {c: role for c_arr, role in arr_cols for c in c_arr}
        all_cols = [scol for scol, _ in all_cols_and_roles.items()]

        if 'target' in self.__dict__ and self.target is not None:
            sdf = sdf.join(self.target, on=SparkDataset.ID_COLUMN, how='inner')

        sdf = sdf.orderBy(SparkDataset.ID_COLUMN).select(self.target_column, *all_cols)
        data = sdf.toPandas()

        # we do it this way, because scol doesn't have a method to retrive name as a str
        all_roles = {c: all_cols_and_roles[c] for c in sdf.columns if c != self.target_column}

        df = pd.DataFrame(data=data.to_dict())
        target_series = df[self.target_column]
        df = df.drop(self.target_column, 1)

        return df, target_series, all_roles

    def set_data(self,
                 data: SparkDataFrame,
                 features: List[str],
                 roles: NpRoles = None,
                 dependencies: Optional[List['SparkDataset']] = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `ignored, always None. just for same interface.
            roles: Dict with roles.
            dependencies: spark dataframes that should be uncached when this spark dataframe has been materialized
        """
        self._validate_dataframe(data)
        super().set_data(data, None, roles)

        if dependencies is not None:
            self._dependencies = dependencies

    def to_pandas(self) -> PandasDataset:
        data, target_data, roles = self._materialize_to_pandas()

        task = Task(self.task.name)
        pds = PandasDataset(data=data, roles=roles, task=task, target=target_data)

        return pds

    def to_numpy(self) -> NumpyDataset:
        data, target_data, roles = self._materialize_to_pandas()

        try:
            target = self.target
            if isinstance(target, pd.Series):
                target = target.to_numpy()
            elif isinstance(target, SparkDataFrame):
                target = target.toPandas().to_numpy()
        except AttributeError:
            target = None

        try:
            folds = self.folds
            if isinstance(folds, pd.Series):
                folds = folds.to_numpy()
            elif isinstance(folds, SparkDataFrame):
                folds = folds.toPandas().to_numpy()
        except AttributeError:
            folds = None

        return NumpyDataset(
            data=data.to_numpy(),
            features=list(data.columns),
            roles=roles,
            task=self.task,
            target=target,
            folds=folds
        )

    @staticmethod
    def _hstack(datasets: Sequence[Any]) -> Any:
        # TODO: we need to use join to implement this operation
        # TODO: to perform join correctly we need to have unique id for each row
        # TODO: and optionally have bucketed tables (bucketed by the same way)
        raise NotImplementedError("It is not yet ready")

    @staticmethod
    def _get_cols(data, k: IntIdx) -> Any:
        raise NotImplementedError("It is not yet ready")

    @staticmethod
    def from_lama(dataset: Union[NumpyDataset, PandasDataset], spark: SparkSession) -> "SparkDataset":
        dataset: PandasDataset = dataset.to_pandas()
        pdf = cast(pd.DataFrame, dataset.data)
        pdf = pdf.copy()
        pdf[SparkDataset.ID_COLUMN] = pdf.index
        sdf = spark.createDataFrame(data=pdf)

        try:
            target = dataset.target
            if isinstance(target, pd.Series):
                target = target.to_numpy()
        except AttributeError:
            target = None

        try:
            folds = dataset.folds
            if isinstance(folds, pd.Series):
                folds = folds.to_numpy()
        except AttributeError:
            folds = None

        return SparkDataset(
            data=sdf,
            roles=dataset.roles,
            task=dataset.task,
            dependencies=[],
            target=target,
            folds=folds
        )

    @staticmethod
    def from_dataset(dataset: "LAMLDataset") -> "LAMLDataset":
        assert isinstance(dataset, SparkDataset), "Can only convert from SparkDataset"
        return dataset
