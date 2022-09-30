"""Base classes for MLPipeline."""
import uuid
import warnings
from copy import copy
from typing import List, cast, Sequence, Union, Tuple, Optional

from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.ml.base import MLPipeline as LAMAMLPipeline
from lightautoml.pipelines.selection.base import EmptySelector
from pyspark.ml import Transformer, PipelineModel

from ..features.base import SparkFeaturesPipeline, SparkEmptyFeaturePipeline
from ..selection.base import SparkSelectionPipelineWrapper
from ...dataset.base import LAMLDataset, SparkDataset
from ...ml_algo.base import SparkTabularMLAlgo
from ...transformers.base import ColumnsSelectorTransformer
from ...utils import Cacher
from ...validation.base import SparkBaseTrainValidIterator


class SparkMLPipeline(LAMAMLPipeline):
    """Spark version of :class:`~lightautoml.pipelines.ml.base.MLPipeline`. Single ML pipeline.

    Merge together stage of building ML model
    (every step, excluding model training, is optional):

        - Pre selection: select features from input data.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Features generation: build new features from selected.
          Performed by
          :class:`~sparklightautoml.pipelines.features.base.SparkFeaturesPipeline`.
        - Post selection: One more selection step - from created features.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Hyperparams optimization for one or multiple ML models.
          Performed by
          :class:`~lightautoml.ml_algo.tuning.base.ParamsTuner`.
        - Train one or multiple ML models:
          Performed by :class:`~sparklightautoml.ml_algo.base.SparkTabularMLAlgo`.
          This step is the only required for at least 1 model.

    """

    def __init__(
        self,
        cacher_key: str,
        ml_algos: Sequence[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SparkSelectionPipelineWrapper] = None,
        features_pipeline: Optional[SparkFeaturesPipeline] = None,
        post_selection: Optional[SparkSelectionPipelineWrapper] = None,
        name: Optional[str] = None,
    ):
        if features_pipeline is None:
            features_pipeline = SparkEmptyFeaturePipeline()

        if pre_selection is None:
            pre_selection = SparkSelectionPipelineWrapper(EmptySelector())

        if post_selection is None:
            post_selection = SparkSelectionPipelineWrapper(EmptySelector())

        super().__init__(ml_algos, force_calc, pre_selection, features_pipeline, post_selection)

        self._cacher_key = cacher_key
        self._output_features = None
        self._output_roles = None
        self._transformer: Optional[Transformer] = None
        self._name = name if name else str(uuid.uuid4())[:5]
        self.ml_algos: List[SparkTabularMLAlgo] = []
        self.pre_selection = cast(SparkSelectionPipelineWrapper, self.pre_selection)
        self.post_selection = cast(SparkSelectionPipelineWrapper, self.post_selection)
        self.features_pipeline = cast(SparkFeaturesPipeline, self.features_pipeline)

    @property
    def name(self) -> str:
        return self._name

    @property
    def transformer(self):
        assert self._transformer is not None, f"{type(self)} seems to be not fitted"
        return self._transformer

    def fit_predict(self, train_valid: SparkBaseTrainValidIterator) -> LAMLDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """

        # train and apply pre selection
        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        train_valid = train_valid.apply_selector(self.post_selection)

        # TODO: SLAMA join - checkpointing here?

        preds: List[SparkDataset] = []
        for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
            ml_algo = cast(SparkTabularMLAlgo, ml_algo)
            ml_algo, curr_preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
            if ml_algo is not None:
                curr_preds = cast(SparkDataset, curr_preds)
                self.ml_algos.append(ml_algo)
                preds.append(curr_preds)
            else:
                warnings.warn(
                    "Current ml_algo has not been trained by some reason. " "Check logs for more details.",
                    RuntimeWarning,
                )

        assert (
            len(self.ml_algos) > 0
        ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

        del self._ml_algos

        val_preds_ds = SparkDataset.concatenate(preds)

        self._transformer = PipelineModel(stages=[
            self.pre_selection.transformer,
            self.features_pipeline.transformer,
            self.post_selection.transformer,
            *[ml_algo.transformer for ml_algo in self.ml_algos]
        ])

        return val_preds_ds

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        out_sdf = self.transformer.transform(dataset.data)

        out_roles = copy(dataset.roles)
        out_roles.update(self.output_roles)

        out_ds = dataset.empty()
        out_ds.set_data(out_sdf, list(out_roles.keys()), out_roles)

        return out_ds
