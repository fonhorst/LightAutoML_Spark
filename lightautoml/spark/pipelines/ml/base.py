"""Base classes for MLPipeline."""
from copy import copy
from typing import List, cast, Sequence, Union, Tuple, Optional

from pyspark.ml import Transformer, PipelineModel

from lightautoml.validation.base import TrainValidIterator
from ..features.base import SparkFeaturesPipeline
from ...dataset.base import LAMLDataset, SparkDataset
from ...ml_algo.base import SparkTabularMLAlgo
from ....ml_algo.tuning.base import ParamsTuner
from ....ml_algo.utils import tune_and_fit_predict
from ....pipelines.features.base import FeaturesPipeline
from ....pipelines.ml.base import MLPipeline as LAMAMLPipeline
from ....pipelines.selection.base import SelectionPipeline


class SparkMLPipeline(LAMAMLPipeline):
    def __init__(
        self,
        ml_algos: Sequence[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SelectionPipeline] = None,
        features_pipeline: Optional[SparkFeaturesPipeline] = None,
        post_selection: Optional[SelectionPipeline] = None,
    ):
        super().__init__(ml_algos, force_calc, pre_selection, features_pipeline, post_selection)
        self._transformer: Optional[Transformer] = None
        self.ml_algos: List[SparkTabularMLAlgo] = []

    @property
    def transformer(self):
        assert self._transformer is not None, f"{type(self)} seems to be not fitted"

        return self._transformer

    def fit_predict(self, train_valid: TrainValidIterator) -> LAMLDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """
        train_valid = copy(train_valid)

        # train and apply pre selection
        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        # with cast(SparkDataset, train_valid.train).applying_temporary_caching():
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        # with cast(SparkDataset, train_valid.train).applying_temporary_caching():
        train_valid = train_valid.apply_selector(self.post_selection)

        for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
            ml_algo = cast(SparkTabularMLAlgo, ml_algo)
            ml_algo, preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
            if ml_algo is not None:
                self.ml_algos.append(ml_algo)
                preds = cast(SparkDataset, preds)
                # TODO: recreate train_valid iterator with preds as train cause we only add new elements there
            else:
                # TODO: warning
                pass

        assert (
            len(self.ml_algos) > 0
        ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

        del self._ml_algos

        # TODO: build pipeline_model
        stages = []
        if self.pre_selection:
            # TODO: cast
            pre_sel = None
            stages.append(pre_sel.transformer)
        if self.features_pipeline:
            sfp = cast(SparkFeaturesPipeline, self.features_pipeline)
            stages.append(sfp.transformer)
        if self.post_selection:
            # TODO: cast
            post_sel = None
            stages.append(post_sel.transformer)

        ml_algo_transformers = [ml_algo.transformer for ml_algo in self.ml_algos]

        self._transformer = PipelineModel(stages=stages + ml_algo_transformers)

        return train_valid.train

    def predict(self, dataset: LAMLDataset) -> LAMLDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        dataset = self.pre_selection.select(dataset)
        dataset = self.features_pipeline.transform(dataset)
        dataset = self.post_selection.select(dataset)

        predictions: List[SparkDataset] = []

        dataset = cast(SparkDataset, dataset)

        # TODO: SPARK-LAMA same problem with caching - we don't know when to uncache
        # we should uncache only after the whole AutoML workflow is materialized
        dataset.cache()

        for model in self.ml_algos:
            pred = cast(SparkDataset, model.predict(dataset))
            predictions.append(pred)

        result = SparkDataset.concatenate(predictions)

        return result
