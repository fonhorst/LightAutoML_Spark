from typing import Union, Tuple, Optional, Sequence

from lightautoml.ml_algo.tuning.base import ParamsTuner, DefaultTuner
from lightautoml.pipelines.ml.nested_ml_pipe import (
    NestedTabularMLPipeline as LAMANestedTabularMLPipeline,
    NestedTabularMLAlgo,
)

from sparklightautoml.dataset.caching import CacheManager
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.pipelines.selection.base import SparkSelectionPipelineWrapper


class SparkNestedTabularMLPipeline(SparkMLPipeline, LAMANestedTabularMLPipeline):
    """
    Same as NestedTabularMLPipeline of LAMA, but redefines a couple of methods via SparkMLPipelineMixin
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        ml_algos: Sequence[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SparkSelectionPipelineWrapper] = None,
        features_pipeline: Optional[SparkFeaturesPipeline] = None,
        post_selection: Optional[SparkSelectionPipelineWrapper] = None,
        cv: int = 1,
        n_folds: Optional[int] = None,
        inner_tune: bool = False,
        refit_tuner: bool = False,
    ):
        if cv > 1:
            new_ml_algos = []

            for n, mt_pair in enumerate(ml_algos):

                try:
                    mod, tuner = mt_pair
                except (TypeError, ValueError):
                    mod, tuner = mt_pair, DefaultTuner()

                if inner_tune:
                    new_ml_algos.append(NestedTabularMLAlgo(mod, tuner, refit_tuner, cv, n_folds))
                else:
                    new_ml_algos.append((NestedTabularMLAlgo(mod, None, True, cv, n_folds), tuner))

            ml_algos = new_ml_algos
        super().__init__(cache_manager, ml_algos, force_calc, pre_selection, features_pipeline, post_selection)
