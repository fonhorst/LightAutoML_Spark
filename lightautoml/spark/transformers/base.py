from typing import cast, Sequence, List, Set

from lightautoml.dataset.utils import concatenate
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.transformers.base import LAMLTransformer, ColumnsSelector as LAMAColumnsSelector, \
    ChangeRoles as LAMAChangeRoles


class SparkTransformer(LAMLTransformer):

    _features = []

    _can_unwind_parents: bool = True

    def fit(self, dataset: SparkDataset) -> "SparkTransformer":

        self._features = dataset.features
        for check_func in self._fit_checks:
            check_func(dataset)

        return self._fit(dataset)

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        for check_func in self._transform_checks:
            check_func(dataset)

        return self._transform(dataset)

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        # TODO: SPARK-LAMA probably we should assume
        #  that fit_transform executes with cache by default
        #  e.g fit_transform returns a cached and materialized dataset
        dataset.cache()
        self.fit(dataset)

        # when True, it means that during fit operation we conducted some action that
        # materialized our current dataset and thus we can unpersist all its dependencies
        # because we have data to propagate in the cache already
        if self._can_unwind_parents:
            dataset.unwind_dependencies()
            deps = [dataset]
        else:
            deps = dataset.dependencies

        result = self.transform(dataset)
        result.dependencies = deps

        return result

    def print_structure(self, indent: str = "") -> str:
        content = ''
        if "transformer_list" in self.__dict__:
            content = '\n'.join([tr.print_structure(indent + "  ") for tr in self.transformer_list])
        name = self._fname_prefix if self._fname_prefix else str(type(self))

        return f"{indent}{name}\n{content}\n" if len(content) > 0 else f"{indent}{name}"

    def print_tr_types(self) -> Set[str]:
        name = self._fname_prefix if self._fname_prefix else str(type(self))

        trs = {name}
        if "transformer_list" in self.__dict__:
            for tr in self.transformer_list:
                trs.update(tr.print_tr_types())

        return trs


class SequentialTransformer(SparkTransformer):
    """
    Transformer that contains the list of transformers and apply one by one sequentially.
    """
    _fname_prefix = "seq"

    def __init__(self, transformer_list: Sequence[SparkTransformer], is_already_fitted: bool = False):
        """

        Args:
            transformer_list: Sequence of transformers.

        """
        self.transformer_list = transformer_list
        self._is_fitted = is_already_fitted

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        """Fit not supported. Needs output to fit next transformer.

        Args:
            dataset: Dataset to fit.

        """
        raise NotImplementedError("Sequential supports only fit_transform.")

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Apply the sequence of transformers to dataset one over output of previous.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        for trf in self.transformer_list:
            dataset = trf.transform(dataset)

        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Sequential ``.fit_transform``.

         Output features - features from last transformer with no prefix.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        if not self._is_fitted:
            for trf in self.transformer_list:
                dataset = trf.fit_transform(dataset)
        else:
            dataset = self.transform(dataset)

        self.features = self.transformer_list[-1].features
        return dataset


class UnionTransformer(SparkTransformer):
    """Transformer that apply the sequence on transformers in parallel on dataset and concatenate the result."""

    _fname_prefix = "union"

    def __init__(self, transformer_list: Sequence[SparkTransformer], n_jobs: int = 1):
        """

        Args:
            transformer_list: Sequence of transformers.
            n_jobs: Number of processes to run fit and transform.

        """
        # TODO: Add multiprocessing version here
        self.transformer_list = [x for x in transformer_list if x is not None]
        self.n_jobs = n_jobs

        assert len(self.transformer_list) > 0, "The list of transformers cannot be empty or contains only None-s"

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        fnames = []

        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                trf.fit(dataset)
                fnames.append(trf.features)

        self.features = fnames

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        res = []

        for trf in self.transformer_list:
            ds = trf.transform(dataset)
            res.append(ds)

        union_res = cast(SparkDataset, concatenate(res))

        return union_res

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Fit and transform transformers in parallel.
         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit and transform on.

        Returns:
            Dataset with new features.

        """
        res = []
        actual_transformers = []

        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                ds = trf.fit_transform(dataset)
                # if ds:
                res.append(ds)
                actual_transformers.append(trf)

        # this concatenate operations also propagates all dependencies
        result = SparkDataset.concatenate(res) if len(res) > 0 else None

        self.transformer_list = actual_transformers
        self.features = result.features

        return result


class ColumnsSelector(LAMAColumnsSelector, SparkTransformer):
    _fname_prefix = "colsel"
    _can_unwind_parents = False


class ChangeRoles(LAMAChangeRoles, SparkTransformer):
    _fname_prefix = "changeroles"
    _can_unwind_parents = False

